# Project Curious Network
# this script fetch a pre-trained model and run series of "classifications" on the sections of the image, trying to
# detect new elements in the environment. The results can then be used to quantify the model's performance

import argparse
import os
import numpy as np
import torch

from models import create_model
from curious_dataset import CuriousDataset

def main():
    parser = argparse.ArgumentParser(description='Train Curious Network')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('-i', '--input_path', type=str, help='Choose the input database path', required=True)
    parser.add_argument('-o', '--output_path', type=str, help='Choose the output path', required=True)

    parser.add_argument('-t', '--type', choices=['cnn_autoencoder', 'cnn_vae', 'vgg16_backend_autoencoder', 'small_cnn'],
                        help='Choose the network type', required=True)
    parser.add_argument('-w', '--weight_path', type=str, help='Choose the pth file path', required=True)
    parser.add_argument('--threshold', type=float, help='Choose the pth file path', required=True)

    # cnn_autoencoder arguments
    parser.add_argument('--cnn_autoencoder_starting_feature_map',
                        type=int, help='Choose the number of starting feature maps for the auto encoder',
                        default=4)
    parser.add_argument('--cnn_autoencoder_growth_factor',
                        type=int, help='Choose the basis of the coefficient by which the feature'
                                       'maps are goind to be multiplied from a layer to the next', default=2)
    parser.add_argument('--cnn_autoencoder_kernel_size', type=int, help='Choose the starting kernel size', default=3)

    # cnn_vae arguments
    parser.add_argument('--cnn_vae_starting_feature_map',
                        type=int, help='Choose the number of starting feature maps for the auto encoder',
                        default=4)
    parser.add_argument('--cnn_vae_growth_factor',
                        type=int, help='Choose the basis of the coefficient by which the feature'
                                       'maps are goind to be multiplied from a layer to the next', default=2)
    parser.add_argument('--cnn_vae_kernel_size', type=int, help='Choose the starting kernel size', default=3)

    # vgg16_backend_autoencoder arguments
    parser.add_argument('--vgg16_backend_autoencoder_train_backend', action='store_true', help='Train the backend')

    # small_cnn arguments
    parser.add_argument('--small_cnn_kernel_size', type=int, help='Set the value for small_cnn', default=3)
    parser.add_argument('--small_cnn_first_output_channels', type=int, help='Set the value for small_cnn', default=8)
    parser.add_argument('--small_cnn_growth_rate', type=int, help='Set the value for small_cnn', default=2)

    args = parser.parse_args()
    run(args)

def run(args):
    model, normalization, roc_curve_thresholds = create_model(args.type, args)
    dataset = CuriousDataset(args.input_path, normalization=normalization)

    os.makedirs(args.output_path, exist_ok=True)

    if torch.cuda.is_available() and args.use_gpu:
        model = model.cuda()

    # Fetchs and prepare the pre-trained model
    model.load_state_dict(torch.load(args.weight_path))
    model.eval()

    for idx, (image, _) in enumerate(dataset):
        print(idx + 1, " /", len(dataset))

        if torch.cuda.is_available() and args.use_gpu:
            image = image.cuda()

        if torch.cuda.is_available() and args.use_gpu:
            error = model(image.unsqueeze(0)).cpu().detach().numpy()
        else:
            error = model(image.unsqueeze(0)).detach().numpy()
        error = np.sqrt(error)
        predicted_roi = error > args.threshold

        output_file = os.path.join(args.output_path, str.replace(dataset.get_filename(idx), '.jpg', '.txt'))
        np.savetxt(output_file, predicted_roi[0], delimiter=',', fmt='%f')

if __name__ == '__main__':
    main()
