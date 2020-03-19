import random
from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision.transforms.functional as F
import numpy as np


class DataAugmentationTransform(object):

    def __init__(self, noise_std_dev=0.001, prob_flip=0.5):
        self._color_jitter = transforms.ColorJitter(brightness=0.4,
                                                    contrast=0.4,
                                                    saturation=0.4,
                                                    hue=0.4)
        self._noise_std_dev = noise_std_dev
        self._prob_flip = prob_flip

    def __call__(self, input):
        image, annotation = input
        image = self._color_jitter(image)
        image, annotation = self._add_noise((image, annotation))
        image, annotation = self._flip((image, annotation))
        return image, annotation

    def _flip(self, input):
        image, annotation = input
        if self.prob_flip > np.random.rand(1)[0]:
            image = image.flip(2)
            annotation = annotation.flip(2)
        return image, annotation

    def _add_noise(self, input):
        image, annotation = input
        image += self._noise_std_dev * torch.randn(image.size())
        return image, annotation



