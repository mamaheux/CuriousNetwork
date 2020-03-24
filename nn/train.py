import argparse
import os
import numpy as np

import torch
from torch.autograd import Variable

import torchvision.transforms as transforms

from barbar import Bar

from curious_dataset import create_dataset_loader
import data_augmentation_transform

from learning_curves import LearningCurves
from metrics.validation_loss import ValidationLoss
from metrics.roc_curve import RocCurve
from metrics.model_execution_time import ModelExecutionTime

from models.cnn_autoencoder import CnnAutoencoder
from models.vgg16_backend_autoencoder import Vgg16BackendAutoencoder
from models.small_cnn import SmallCnnWithAutoencoder

def main():
    parser = argparse.ArgumentParser(description='Train Curious Network')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--train_path', type=str, help='Choose the training database path', required=True)
    parser.add_argument('--val_path', type=str, help='Choose the validation database path', required=True)
    parser.add_argument('--test_path', type=str, help='Choose the validation database path', required=True)
    parser.add_argument('-o', '--output_path', type=str, help='Choose the output path', required=True)
    parser.add_argument('-n', '--name', type=str, help='Choose the model name', required=True)

    parser.add_argument('-t', '--type', choices=['cnn_autoencoder', 'vgg16_backend_autoencoder', 'small_cnn'],
                        help='Choose the network type', required=True)
    parser.add_argument('-s', '--batch_size', type=int, help='Set the batch size for the training', default=20)
    parser.add_argument('-d', '--data_augmentation', action='store_true', help='Use data augmentation or not')
    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)
    parser.add_argument('--weight_decay', type=float, help='Choose the weight decay', required=True)

    # small_cnn arguments
    parser.add_argument('--small_cnn_kernel_size', type=int, help='Set the value for small_cnn', default=3)
    parser.add_argument('--small_cnn_first_output_channels', type=int, help='Set the value for small_cnn', default=8)
    parser.add_argument('--small_cnn_growth_rate', type=int, help='Set the value for small_cnn', default=2)

    # Parameters of the CNN autoencoder
    parser.add_argument('--cnn_autoencoder_starting_feature_map',
                        type=int, help='Choose the number of starting feature maps for the auto encoder',
                        default=4)
    parser.add_argument('--cnn_autoencoder_growth_factor',
                        type=int, help='Choose the basis of the coefficient by which the feature' 
                                       'maps are goind to be multiplied from a layer to the next', default=2)
    parser.add_argument('--cnn_autoencoder_kernel_size', type=int, help='Choose the starting kernel size', default=3)

    args = parser.parse_args()
    train(args)

def train(args):
    if args.data_augmentation:
        transform = data_augmentation_transform.DataAugmentationTransform()
    else:
        transform = None
    dataset_loader = create_dataset_loader(args.train_path, args.batch_size, transform=transform,
                                           normalization=transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225]))
    model, roc_curve_thresholds = create_model(args.type, args)

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
                image = Variable(image).cuda()
            else:
                image = Variable(image)

            output = model(image)
            loss = output.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(dataset_loader)
        learning_curves.add_training_loss_value(train_loss)
        print('training loss: {}'.format(train_loss))

        model.eval()
        validation_loss = ValidationLoss(args.val_path, model).calculate()
        learning_curves.add_validation_loss_value(validation_loss)
        print('validation loss: {}'.format(validation_loss))

        roc_curve_rates = RocCurve(args.test_path, model, roc_curve_thresholds).calculate()

        name_with_epoch = args.name + '_epoch_{}'.format(epoch)
        torch.save(model.state_dict(), os.path.join(args.output_path, name_with_epoch + '.pth'))
        np.savetxt(os.path.join(args.output_path, name_with_epoch + '_val.txt'), np.array([validation_loss]), delimiter=',', fmt='%f')
        np.savetxt(os.path.join(args.output_path, name_with_epoch + '_roc.txt'), roc_curve_rates, delimiter=',', fmt='%f')

    learning_curves.save_figure(os.path.join(args.output_path, args.name + '_learning_curves.png'))

    model_execution_time = ModelExecutionTime(args.test_path, model)
    with open(os.path.join(args.output_path, args.name + '_execution_time.txt'), "w") as text_file:
        text_file.write("Forward: {} seconds\nBackward: {} seconds".format(*model_execution_time.calculate()))

def create_model(type, hyperparameters):
    if type == 'cnn_autoencoder':
        roc_curve_thresholds = np.linspace(0, 1, num=1000)
        return CnnAutoencoder(), roc_curve_thresholds
    elif type == 'vgg16_backend_autoencoder':
        roc_curve_thresholds = np.linspace(0, 10, num=10000)
        return Vgg16BackendAutoencoder(), roc_curve_thresholds
    elif type == 'small_cnn':
        roc_curve_thresholds = np.linspace(0, 10, num=10000)
        return SmallCnnWithAutoencoder(), roc_curve_thresholds
    else:
        raise ValueError('Invalid model type')

if __name__ == '__main__':
    main()
