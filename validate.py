import argparse
from datasets import create_dataset
from utils import parse_configuration
from models import create_model
import os
from utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import numpy as np

"""Performs validation of a specified model.

Input params:
    config_file: Either a string with the path to the JSON
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and
        model-specific settings.
"""
def validate(config_file):
    print('Reading config file...')
    configuration = parse_configuration(config_file)

    print('Initializing dataset...')
    val_dataset = create_dataset(configuration['val_dataset_params'])
    val_dataset_size = len(val_dataset)
    print('The number of validation samples = {0}'.format(val_dataset_size))

    print('Initializing model...')
    model = create_model(configuration['model_params'])
    model.setup()
    model.eval()

    print('Initializing visualization...')
    visualizer = Visualizer(configuration['visualization_params_validation'])   # create a visualizer that displays images and plots

    model.pre_epoch_callback(configuration['model_params']['load_checkpoint'])

    #Loops through all validation data and runs though model
    for i, data in enumerate(val_dataset):
        if (i > 0):
            quit()
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference

        input = model.input_real[0].permute(1, 2, 0).cpu().detach().numpy()
        img = model.output_img[0].permute(1, 2, 0).cpu().detach().numpy()
        img_filtered = model.output_filtered[0].permute(1, 2, 0).cpu().detach().numpy()

        print(np.min(img), np.max(img))
        print(np.min(img_filtered), np.max(img_filtered))

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(input)
        axs[1].imshow(img)
        axs[2].imshow(img_filtered)
        plt.show()

    #Where results are calculated and visualized
    model.post_epoch_callback(configuration['model_params']['load_checkpoint'], visualizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    validate(args.configfile)
