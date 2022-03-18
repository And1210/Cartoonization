import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as nf
from torch.nn.utils import spectral_norm
from models.base_model import BaseModel
from losses.LSGanLoss import LSGanLoss
from losses.VariationLoss import VariationLoss
from models.pretrained import VGGCaffePreTrained
from functools import partial
from typing import Tuple
from utils.superpix import slic, adaptive_slic, sscolor
import itertools
import numpy as np
from joblib import Parallel, delayed

def simple_superpixel(batch_image: np.ndarray, superpixel_fn: callable) -> np.ndarray:
  """ convert batch image to superpixel
  Args:
      batch_image (np.ndarray): np.ndarry, shape must be [b,h,w,c]
      seg_num (int, optional): . Defaults to 200.
  Returns:
      np.ndarray: superpixel array, shape = [b,h,w,c]
  """
  num_job = batch_image.shape[0]
  batch_out = Parallel(n_jobs=num_job)(delayed(superpixel_fn)
                                       (image) for image in batch_image)
  return np.array(batch_out)

class GuidedFilter(nn.Module):
  def box_filter(self, x: torch.Tensor, r):
    ch = x.shape[1]
    k = 2 * r + 1
    weight = 1 / ((k)**2)  # 1/9
    # [c,1,3,3] * 1/9
    box_kernel = torch.ones((ch, 1, k, k), dtype=torch.float32, device=x.device).fill_(weight)
    # same padding
    return nf.conv2d(x, box_kernel, padding=r, groups=ch)

  def forward(self, x: torch.Tensor, y: torch.Tensor, r, eps=1e-2):
    b, c, h, w = x.shape
    device = x.device
    # 全1的图像进行滤波的结果
    N = self.box_filter(torch.ones((1, 1, h, w), dtype=x.dtype, device=device), r)

    mean_x = self.box_filter(x, r) / N
    mean_y = self.box_filter(y, r) / N
    cov_xy = self.box_filter(x * y, r) / N - mean_x * mean_y
    var_x = self.box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = self.box_filter(A, r) / N
    mean_b = self.box_filter(b, r) / N

    output = mean_A * x + mean_b
    return output

class ColorShift(nn.Module):
  def __init__(self, mode='uniform'):
    super().__init__()
    self.dist: Distribution = None
    self.mode = mode

  def setup(self, device: torch.device):
    # NOTE 原论文输入的bgr图像，此处需要改为rgb
    if self.mode == 'normal':
      self.dist = torch.distributions.Normal(
          torch.tensor((0.299, 0.587, 0.114), device=device),
          torch.tensor((0.1, 0.1, 0.1), device=device))
    elif self.mode == 'uniform':
      self.dist = torch.distributions.Uniform(
          torch.tensor((0.199, 0.487, 0.014), device=device),
          torch.tensor((0.399, 0.687, 0.214), device=device))

  #Allow taking mutiple images batches as input
  #So we can do: gray_fake, gray_cartoon = ColorShift(output, input_cartoon)
  def forward(self, *image_batches: torch.Tensor) -> Tuple[torch.Tensor]:
    # Sample the random color shift coefficients
    weights = self.dist.sample()

    # images * self.weights[None, :, None, None] => Apply weights to r,g,b channels of each images
    # torch.sum(, dim=1) => Sum along the channels so (B, 3, H, W) become (B, H, W)
    # .unsqueeze(1) => add back the channel so (B, H, W) become (B, 1, H, W)
    # .repeat(1, 3, 1, 1) => (B, 1, H, W) become (B, 3, H, W) again
    return ((((torch.sum(images * weights[None, :, None, None], dim= 1)) / weights.sum()).unsqueeze(1)).repeat(1, 3, 1, 1) for images in image_batches)

class Mean(nn.Module):
  def __init__(self, dim: list, keepdim=False):
    super().__init__()
    self.dim = dim
    self.keepdim = keepdim

  def forward(self, x):
    return torch.mean(x, self.dim, self.keepdim)

def calc(pad, h, k, s):
  import math
  return math.floor((h + 2 * pad - (k - 1) - 1) / s + 1)

class ResBlock(nn.Module):
  def __init__(self, in_channel, out_channel=32):
    super().__init__()
    self.conv = nn.Conv2d(in_channel, out_channel, [3, 3], padding=1)
    self.conv1 = nn.Conv2d(out_channel, out_channel, [3, 3], padding=1)
    self.leaky_relu = nn.LeakyReLU(inplace=True)

  def forward(self, inputs):
    x = self.conv1(self.leaky_relu(self.conv1(inputs)))
    return x + inputs


class UnetGenerator(nn.Module):
  def __init__(self, channel=32, num_blocks=4):
    super().__init__()

    self.conv = nn.Conv2d(3, channel, [7, 7], padding=3)  # same 256,256
    self.conv1 = nn.Conv2d(channel, channel, [3, 3], stride=2, padding=1)  # same 128,128
    self.conv2 = nn.Conv2d(channel, channel * 2, [3, 3], padding=1)  # 128,128
    self.conv3 = nn.Conv2d(channel * 2, channel * 2, [3, 3], stride=2, padding=1)  # 64,64
    self.conv4 = nn.Conv2d(channel * 2, channel * 4, [3, 3], padding=1)  # 64,64

    self.resblock = nn.Sequential(*[ResBlock(channel * 4, channel * 4) for i in range(num_blocks)])

    self.conv5 = nn.Conv2d(channel * 4, channel * 2, [3, 3], padding=1)  # 64,64
    self.conv6 = nn.Conv2d(channel * 2, channel * 2, [3, 3], padding=1)  # 64,64
    self.conv7 = nn.Conv2d(channel * 2, channel, [3, 3], padding=1)  # 64,64
    self.conv8 = nn.Conv2d(channel, channel, [3, 3], padding=1)  # 64,64
    self.conv9 = nn.Conv2d(channel, 3, [7, 7], padding=3)  # 64,64

    self.leak_relu = nn.LeakyReLU(inplace=True)
    self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
    self.act = nn.Tanh()

  def forward(self, inputs):
    x0 = self.conv(inputs)
    x0 = self.leak_relu(x0)  # 256, 256, 32

    x1 = self.conv1(x0)
    x1 = self.leak_relu(x1)
    x1 = self.conv2(x1)
    x1 = self.leak_relu(x1)  # 128, 128, 64

    x2 = self.conv3(x1)
    x2 = self.leak_relu(x2)
    x2 = self.conv4(x2)
    x2 = self.leak_relu(x2)  # 64, 64, 128

    x2 = self.resblock(x2)  # 64, 64, 128
    x2 = self.conv5(x2)
    x2 = self.leak_relu(x2)  # 64, 64, 64

    x3 = self.upsample(x2)
    x3 = self.conv6(x3 + x1)
    x3 = self.leak_relu(x3)
    x3 = self.conv7(x3)
    x3 = self.leak_relu(x3)  # 128, 128, 32

    x4 = self.upsample(x3)
    x4 = self.conv8(x4 + x0)
    x4 = self.leak_relu(x4)
    x4 = self.conv9(x4)  # 256, 256, 32

    return self.act(x4)


class SpectNormDiscriminator(nn.Module):
  def __init__(self, channel=32, patch=True):
    super().__init__()
    self.channel = channel
    self.patch = patch
    in_channel = 3
    l = []
    for idx in range(3):
      l.extend([
          spectral_norm(nn.Conv2d(in_channel, channel * (2**idx), 3, stride=2, padding=1)),
          nn.LeakyReLU(inplace=True),
          spectral_norm(nn.Conv2d(channel * (2**idx), channel * (2**idx), 3, stride=1, padding=1)),
          nn.LeakyReLU(inplace=True),
      ])
      in_channel = channel * (2**idx)
    self.body = nn.Sequential(*l)
    if self.patch:
      self.head = spectral_norm(nn.Conv2d(in_channel, 1, 1, padding=0))
    else:
      self.head = nn.Sequential(Mean([1, 2]), nn.Linear(in_channel, 1))

  def forward(self, x):
    x = self.body(x)
    x = self.head(x)
    return x

#PyTorch defined model
class Cartoonization(nn.Module):
    SuperPixelDict = {
          'slic': slic,
          'adaptive_slic': adaptive_slic,
          'sscolor': sscolor}

    def __init__(self, device, superpixel_fn='sscolor', superpixel_kwarg={'seg_num': 200}):
        super(Cartoonization, self).__init__()

        self.device = device

        self.generator = UnetGenerator()
        self.disc_gray = SpectNormDiscriminator()
        self.disc_blur = SpectNormDiscriminator()
        self.guided_filter = GuidedFilter()
        self.colorshift = ColorShift()
        self.pretrained = VGGCaffePreTrained()
        self.superpixel_fn = partial(self.SuperPixelDict[superpixel_fn],
                                     **superpixel_kwarg)

        self.colorshift.setup(self.device)
        self.pretrained.setup(self.device)

    #The function which feeds forwards data into different layers
    #Use the above defined layers here on input data
    def generator_forward(self, x):
        input_cartoon, input_photo = x

        generator_img = self.generator(input_photo)
        output = self.guided_filter(input_photo, generator_img, r=1)

        # 1. blur for Surface Representation
        blur_fake = self.guided_filter(output, output, r=5, eps=2e-1)
        blur_fake_logit = self.disc_blur(blur_fake)

        # 2. gray for Textural Representation
        gray_fake, = self.colorshift(output)
        gray_fake_logit = self.disc_gray(gray_fake)

        # 3. superpixel for Structure Representation
        input_superpixel = torch.from_numpy(
            simple_superpixel(output.detach().permute((0, 2, 3, 1)).cpu().numpy(),
                                self.superpixel_fn)
                                ).to(self.device).permute((0, 3, 1, 2))

        vgg_output = self.pretrained(output)
        _, c, h, w = vgg_output.shape
        vgg_superpixel = self.pretrained(input_superpixel)

        # 4. Content
        vgg_photo = self.pretrained(input_photo)

        g_imgs = (blur_fake, gray_fake, input_superpixel)

        return output, blur_fake_logit, gray_fake_logit, vgg_output, vgg_superpixel, vgg_photo, (c, h, w), g_imgs

    def discriminator_forward(self, x):
        input_cartoon, input_photo = x

        generator_img = self.generator(input_photo)
        output = self.guided_filter(input_photo, generator_img, r=1)

        # 1. blur for Surface Representation
        blur_fake = self.guided_filter(output, output, r=5, eps=2e-1)
        blur_cartoon = self.guided_filter(input_cartoon, input_cartoon, r=5, eps=2e-1)
        blur_real_logit = self.disc_blur(blur_cartoon)
        blur_fake_logit = self.disc_blur(blur_fake)

        # 2. gray for Textural Representation
        gray_fake, gray_cartoon = self.colorshift(output, input_cartoon)
        gray_real_logit = self.disc_gray(gray_cartoon)
        gray_fake_logit = self.disc_gray(gray_fake)

        d_imgs = (blur_fake, blur_cartoon, gray_fake, gray_cartoon)

        return output, blur_real_logit, blur_fake_logit, gray_real_logit, gray_fake_logit, d_imgs

    # def train_forward(self, input):
    #     input_cartoon, input_photo = input
    #
    #     return self.generator_forward(input, generator_img, output, blur_fake), self.discriminator_forward(input, generator_img, output, blur_fake)

    def forward(self, input_photo):
        generator_img = self.generator(input_photo)
        output = self.guided_filter(input_photo, generator_img, r=1, eps=5e-3)
        return generator_img, output

#The abstract model class, uses above defined class and is used in the train script
class Cartoonizationmodel(BaseModel):
    """basenet for fer2013"""

    def __init__(self, configuration):
        super().__init__(configuration)

        #Initialize model defined above
        self.model = Cartoonization(self.device)
        self.model.cuda()

        # torch.autograd.set_detect_anomaly(True)

        #Define loss function
        self.criterion_loss = nn.CrossEntropyLoss().cuda()
        self.lsgan_loss = LSGanLoss()
        self.l1_loss = nn.L1Loss('mean')
        self.variation_loss = VariationLoss(1)

        #Define optimizer
        self.optimizer_g = torch.optim.Adam(
            self.model.generator.parameters(),
            lr=configuration['lr_g'],
            betas=configuration['betas']
        )
        self.optimizer_d = torch.optim.Adam(
            itertools.chain(self.model.disc_blur.parameters(),
                            self.model.disc_gray.parameters()),
            lr=configuration['lr_d'],
            betas=configuration['betas']
        )

        #Need to include these arrays with the optimizers and names of loss functions and models
        #Will be used by other functions for saving/loading
        self.optimizers = [self.optimizer_g, self.optimizer_d]
        self.loss_names = ['g_total', 'd_total', 'pretrain']
        self.network_names = ['model']

        self.loss_g_total = 0
        self.loss_d_total = 0
        self.loss_pretrain = 0

        self.val_images = []
        self.val_predictions = []
        self.val_labels = []

    #Calls the models forwards function
    def forward(self):
        x = (self.input_sim, self.input_real)
        # self.output = self.model.forward(x)

        self.output_img, self.output_filtered = self.model.forward(x[1])

        return self.output_img, self.output_filtered

    def g_forward(self):
        x = (self.input_sim, self.input_real)
        self.g_output, self.g_blur_fake_logit, self.g_gray_fake_logit, self.g_vgg_output, self.g_vgg_superpixel, self.g_vgg_photo, self.g_shape, self.g_imgs = self.model.generator_forward(x)

    def d_forward(self):
        x = (self.input_sim, self.input_real)
        self.d_output, self.d_blur_real_logit, self.d_blur_fake_logit, self.d_gray_real_logit, self.d_gray_fake_logit, self.d_imgs = self.model.discriminator_forward(x)

    def pretrain_forward(self):
        self.output_real, _ = self.model.forward(self.input_real)

    #Computes the loss with the specified name (in this case 'total')
    def compute_loss(self):
        self.compute_g_loss()
        self.compute_d_loss()
        # self.loss_total = self.criterion_loss(self.output, self.label)

    def compute_g_loss(self):
        GH0, GH1, GH2, GH3, GH4 = (1, 0.1, 200, 200, 10)
        self.g_loss_blur = self.lsgan_loss._g_loss(self.g_blur_fake_logit)
        self.g_loss_gray = self.lsgan_loss._g_loss(self.g_gray_fake_logit)
        c, h, w = self.g_shape
        self.g_loss_superpixel = self.l1_loss(self.g_vgg_superpixel, self.g_vgg_output)/(c * h * w)
        self.g_loss_photo = self.l1_loss(self.g_vgg_photo, self.g_vgg_output)/(c * h * w)
        self.g_loss_tv = self.variation_loss(self.g_output)
        self.loss_g_total = GH0*self.g_loss_blur + GH1*self.g_loss_gray + GH2*self.g_loss_superpixel + GH3*self.g_loss_photo + GH4*self.g_loss_tv

    def compute_d_loss(self):
        self.d_loss_blur = self.lsgan_loss._d_loss(self.d_blur_real_logit, self.d_blur_fake_logit)
        self.d_loss_gray = self.lsgan_loss._d_loss(self.d_gray_real_logit, self.d_gray_fake_logit)
        self.loss_d_total = self.d_loss_blur + self.d_loss_gray

    def compute_pretrain_loss(self):
        self.loss_pretrain = self.l1_loss(self.input_real, self.output_real)

    #Compute backpropogation for the model
    def optimize_parameters(self):
        self.loss_g_total.backward()
        self.loss_d_total.backward()
        self.optimizer_g.step()
        self.optimizer_d.step()
        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()
        torch.cuda.empty_cache()

    def optimize_g_parameters(self):
        self.loss_g_total.backward()
        self.optimizer_g.step()
        self.optimizer_g.zero_grad()
        torch.cuda.empty_cache()

    def optimize_d_parameters(self):
        self.loss_d_total.backward()
        self.optimizer_d.step()
        self.optimizer_d.zero_grad()
        torch.cuda.empty_cache()

    def optimize_pretrain_parameters(self):
        self.loss_pretrain.backward()
        self.optimizer_g.step()
        self.optimizer_g.zero_grad()
        torch.cuda.empty_cache()

    #Test function for the model
    def test(self):
        super().test() # run the forward pass

        self.g_forward()
        self.d_forward()
        # save predictions and labels as flat tensors
        # self.val_images.append(self.input)
        # self.val_predictions.append(self.output)
        # self.val_labels.append(self.label)

    #Should be run after each epoch, outputs accuracy
    def post_epoch_callback(self, epoch, visualizer):
        self.val_predictions = torch.cat(self.val_predictions, dim=0)
        predictions = torch.argmax(self.val_predictions, dim=1)
        predictions = torch.flatten(predictions).cpu()

        self.val_labels = torch.cat(self.val_labels, dim=0)
        labels = torch.flatten(self.val_labels).cpu()

        self.val_images = torch.squeeze(torch.cat(self.val_images, dim=0)).cpu()

        # Calculate and show accuracy
        val_accuracy = accuracy_score(labels, predictions)

        metrics = OrderedDict()
        metrics['Accuracy'] = val_accuracy

        if (visualizer != None):
            visualizer.plot_current_validation_metrics(epoch, metrics)
        print('Validation accuracy: {0:.3f}'.format(val_accuracy))

        # Here you may do something else with the validation data such as
        # displaying the validation images or calculating the ROC curve

        self.val_images = []
        self.val_predictions = []
        self.val_labels = []


if __name__ == "__main__":
    net = TEMPLATEmodel().cuda()
    from torchsummary import summary

    print(summary(net, input_size=(1, 48, 48)))
