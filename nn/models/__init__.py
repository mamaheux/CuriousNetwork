import numpy as np

import torchvision.transforms as transforms

from models.cnn_autoencoder import CnnAutoencoder
from models.cnn_vae import CnnVae
from models.vgg16_backend_autoencoder import Vgg16BackendAutoencoder
from models.small_cnn import SmallCnnWithAutoencoder
from models.small_cnn_dense_blocks import SmallCnnWithAutoencoderDenseBlocks

from utils import MinMaxNormalization


def create_model(type, hyperparameters):
    if type == 'cnn_autoencoder':
        roc_curve_thresholds = np.linspace(0, 1, num=1000)
        return CnnAutoencoder(ini_feature_maps=hyperparameters.cnn_autoencoder_starting_feature_map,
                              feature_maps_growth_factor=hyperparameters.cnn_autoencoder_growth_factor,
                              kernel_size=hyperparameters.cnn_autoencoder_kernel_size,
                              useBatchNorm=hyperparameters.use_batchNorm), \
               MinMaxNormalization(), \
               roc_curve_thresholds

    if type == 'cnn_vae':
        roc_curve_thresholds = np.linspace(0, 1, num=1000)
        return CnnVae(ini_feature_maps=hyperparameters.cnn_vae_starting_feature_map,
                      feature_maps_growth_factor=hyperparameters.cnn_vae_growth_factor,
                      kernel_size=hyperparameters.cnn_vae_kernel_size,
                      useBatchNorm=hyperparameters.use_batchNorm), \
               MinMaxNormalization(), \
               roc_curve_thresholds

    elif type == 'vgg16_backend_autoencoder':
        roc_curve_thresholds = np.linspace(0, 10, num=10000)
        return Vgg16BackendAutoencoder(train_backend=hyperparameters.vgg16_backend_autoencoder_train_backend,
                                       useBatchNorm=hyperparameters.use_batchNorm), \
               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), \
               roc_curve_thresholds

    elif type == 'small_cnn':
        roc_curve_thresholds = np.linspace(0, 10, num=10000)
        return SmallCnnWithAutoencoder(kernel_size=hyperparameters.small_cnn_kernel_size,
                                       first_output_channels=hyperparameters.small_cnn_first_output_channels,
                                       growth_rate=hyperparameters.small_cnn_growth_rate,
                                       useBatchNorm=hyperparameters.use_batchNorm), \
               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), \
               roc_curve_thresholds

    elif type == 'small_cnn_dense_blocks':
        roc_curve_thresholds = np.linspace(0, 10, num=10000)
        return SmallCnnWithAutoencoderDenseBlocks(kernel_size=hyperparameters.small_cnn_dense_blocks_kernel_size,
                                                  growth_rate=hyperparameters.small_cnn_dense_blocks_growth_rate,
                                                  useBatchNorm=hyperparameters.use_batchNorm), \
               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), \
               roc_curve_thresholds

    else:
        raise ValueError('Invalid model type')
