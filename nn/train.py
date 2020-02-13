import argparse
import os
import numpy as np

import torch
from torch.autograd import Variable
from barbar import Bar

import curious_dataset
from metrics.validation_loss import ValidationLoss
from metrics.roc_curve import RocCurve
from models.cnn_autoencoder import CnnAutoencoder

BATCH_SIZE = 20
roc_curve_thresholds = np.linspace(0, 3, num=1000)

def main():
    parser = argparse.ArgumentParser(description='Train Curious Network')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--train_path', type=str, help='Choose the training database path', required=True)
    parser.add_argument('--val_path', type=str, help='Choose the validation database path', required=True)
    parser.add_argument('--test_path', type=str, help='Choose the validation database path', required=True)
    parser.add_argument('-o', '--output_path', type=str, help='Choose the output path', required=True)
    parser.add_argument('-n', '--name', type=str, help='Choose the model name', required=True)

    parser.add_argument('-t', '--type', choices=['cnn_autoencoder', 'backend_cnn', 'small_cnn'],
                        help='Choose the network type', required=True)
    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)
    parser.add_argument('--weight_decay', type=float, help='Choose the weight decay', required=True)

    args = parser.parse_args()
    hyperparameters = { 'learning_rate':args.learning_rate,
                        'epoch_count':args.epoch_count,
                        'weight_decay':args.weight_decay }

    train(args.use_gpu,
          args.train_path,
          args.val_path,
          args.test_path,
          args.output_path,
          args.name,
          args.type,
          hyperparameters)

def train(use_gpu, train_path, val_path, test_path, output_path, name, model_type, hyperparameters):
    dataset_loader = curious_dataset.create_dataset_loader(train_path, BATCH_SIZE)
    model = create_model(model_type, hyperparameters)

    if torch.cuda.is_available() and use_gpu:
        model = model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hyperparameters['learning_rate'],
                                 weight_decay=hyperparameters['weight_decay'])

    for epoch in range(hyperparameters['epoch_count']):
        print('epoch [{}/{}]'.format(epoch + 1, hyperparameters['epoch_count']))
        train_loss = 0.0

        for idx, image in enumerate(Bar(dataset_loader)):
            if torch.cuda.is_available() and use_gpu:
                image = Variable(image).cuda()
            else:
                image = Variable(image)

            output = model(image)
            loss = criterion(output, torch.zeros(output.size()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(dataset_loader)
        print('training loss: {:.4f}'.format(train_loss))

    model.eval()
    validation_loss = ValidationLoss(val_path, model).calculate()
    roc_curve_rates = RocCurve(test_path, model, roc_curve_thresholds).calculate()

    print('validation loss: {:.4f}'.format(validation_loss))

    os.makedirs(output_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_path, name + '.pth'))
    np.savetxt(os.path.join(output_path, name + '_val.txt'), np.array([validation_loss]), delimiter=',', fmt='%f')
    np.savetxt(os.path.join(output_path, name + '_roc.txt'), roc_curve_rates, delimiter=',', fmt='%f')

def create_model(type, hyperparameters):
    if type == 'cnn_autoencoder':
        return CnnAutoencoder()
    elif type == 'backend_cnn':
        raise NotImplementedError()
    elif type == 'small_cnn':
        raise NotImplementedError()
    else:
        raise ValueError('Invalid model type')

if __name__ == '__main__':
    main()
