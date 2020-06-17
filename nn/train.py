# Project Curious Network
# Training script

import argparse
import os
import numpy as np

import torch

from barbar import Bar

from curious_dataset import create_dataset_loader
import data_augmentation_transform

from learning_curves import LearningCurves
from metrics.validation_loss import ValidationLoss
from metrics.roc_curve import RocCurve
from metrics.model_execution_time import ModelExecutionTime

from models import create_model

def main():

    # Specify which flags are supported by the script
    parser = argparse.ArgumentParser(description='Train Curious Network')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--train_path', type=str, help='Choose the training database path', required=True)
    parser.add_argument('--val_path', type=str, help='Choose the validation database path', required=True)
    parser.add_argument('--test_path', type=str, help='Choose the validation database path', required=True)
    parser.add_argument('-o', '--output_path', type=str, help='Choose the output path', required=True)
    parser.add_argument('-n', '--name', type=str, help='Choose the model name', required=True)

    parser.add_argument('-t', '--type',
                        choices=['cnn_autoencoder', 'cnn_vae', 'vgg16_backend_autoencoder', 'mnasnet_backend_autoencoder', 'small_cnn',
                                 'small_cnn_dense_blocks'],
                        help='Choose the network type', required=True)
    parser.add_argument('-s', '--batch_size', type=int, help='Set the batch size for the training', default=20)
    parser.add_argument('-d', '--data_augmentation', action='store_true', help='Use data augmentation or not')
    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)
    parser.add_argument('--weight_decay', type=float, help='Choose the weight decay', required=True)

    # cnn_autoencoder flags
    parser.add_argument('--cnn_autoencoder_starting_feature_map',
                        type=int, help='Choose the number of starting feature maps for the autoencoder',
                        default=4)
    parser.add_argument('--cnn_autoencoder_growth_factor',
                        type=int, help='Choose the basis of the coefficient by which the feature'
                                       'maps are goind to be multiplied from a layer to the next', default=2)
    parser.add_argument('--cnn_autoencoder_kernel_size', type=int, help='Choose the starting kernel size', default=3)

    # cnn_vae flags
    parser.add_argument('--cnn_vae_starting_feature_map',
                        type=int, help='Choose the number of starting feature maps for the autoencoder',
                        default=4)
    parser.add_argument('--cnn_vae_growth_factor',
                        type=int, help='Choose the basis of the coefficient by which the feature'
                                       'maps are goind to be multiplied from a layer to the next', default=2)
    parser.add_argument('--cnn_vae_kernel_size', type=int, help='Choose the starting kernel size', default=3)

    # vgg16_backend_autoencoder flags
    parser.add_argument('--vgg16_backend_autoencoder_train_backend', action='store_true', help='Train the backend')

    # mnasnet_backend_autoencoder flags
    parser.add_argument('--mnasnet_backend_autoencoder_train_backend', action='store_true', help='Train the backend')

    # small_cnn flags
    parser.add_argument('--small_cnn_kernel_size', type=int, help='Set the value for small_cnn', default=3)
    parser.add_argument('--small_cnn_first_output_channels', type=int, help='Set the value for small_cnn', default=8)
    parser.add_argument('--small_cnn_growth_rate', type=int, help='Set the value for small_cnn', default=2)

    # small_cnn_dense_blocks flags
    parser.add_argument('--small_cnn_dense_blocks_kernel_size', type=int,
                        help='Set the value for small_cnn_dense_blocks', default=3)
    parser.add_argument('--small_cnn_dense_blocks_growth_rate', type=int,
                        help='Set the value for small_cnn_dense_blocks', default=2)

    args = parser.parse_args()
    train(args)


def train(args):

    # Allows to fetch the VGG16 back-end locally since no internet connexion is
    # available on the calculation machines of Calcul Canada
    os.environ['TORCH_HOME'] = 'models'
    model, normalization, roc_curve_thresholds = create_model(args.type, args)

    if args.data_augmentation:
        transform = data_augmentation_transform.DataAugmentationTransform()
    else:
        transform = None
    dataset_loader = create_dataset_loader(args.train_path, args.batch_size, transform=transform,
                                           normalization=normalization)

    os.makedirs(args.output_path, exist_ok=True)

    if torch.cuda.is_available() and args.use_gpu:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    learning_curves = LearningCurves()
    for epoch in range(args.epoch_count):
        print('epoch [{}/{}]'.format(epoch + 1, args.epoch_count))
        train_loss = 0.0

        model.train()
        for idx, (image, _) in enumerate(Bar(dataset_loader)):
            if torch.cuda.is_available() and args.use_gpu:
                image = image.cuda()

            output = model(image)
            loss = output.sum() + model.internal_loss(use_gpu=args.use_gpu)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(dataset_loader)
        learning_curves.add_training_loss_value(train_loss)
        print('training loss: {}'.format(train_loss))

        model.eval()
        name_with_epoch = args.name + '_epoch_{}'.format(epoch)

        validation_loss = ValidationLoss(args.val_path, normalization, model, roc_curve_thresholds) \
            .calculate(use_gpu=args.use_gpu)
        learning_curves.add_validation_loss_value(validation_loss)
        print('validation loss: {}'.format(validation_loss))
        np.savetxt(os.path.join(args.output_path, name_with_epoch + '_val.txt'), np.array([validation_loss]),
                   delimiter=',', fmt='%f')

        roc_curve = RocCurve(args.test_path, normalization, model, roc_curve_thresholds)
        roc_curve.save(os.path.join(args.output_path, name_with_epoch + '_roc'), use_gpu=args.use_gpu)

        torch.save(model.state_dict(), os.path.join(args.output_path, name_with_epoch + '.pth'))

    learning_curves.save_figure(os.path.join(args.output_path, args.name + '_learning_curves.png'))

    model_execution_time = ModelExecutionTime(args.test_path, normalization, model)
    with open(os.path.join(args.output_path, args.name + '_execution_time.txt'), "w") as text_file:
        text_file.write("Forward: {} seconds\nBackward: {} seconds"
                        .format(*model_execution_time.calculate(use_gpu=args.use_gpu)))

if __name__ == '__main__':
    main()
