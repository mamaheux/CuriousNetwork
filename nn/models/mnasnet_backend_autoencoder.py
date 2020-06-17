# Project Curious Network

import torch
import torch.nn.functional as F

import torchvision.models as models

from models.base_model import BaseModel
import curious_dataset

# Receptive field of the backend : 35x35
class MnasnetBackendAutoencoder(BaseModel):
    def __init__(self, train_backend=False):
        super(MnasnetBackendAutoencoder, self).__init__()

        backend_output_channels = 24

        mnasnet = models.mnasnet1_0(pretrained=True)
        mnasnet_layers = list(mnasnet.layers.children())[:9]
        mnasnet_layers.append(torch.nn.MaxPool2d(4, stride=4))
        mnasnet_layers.append(torch.nn.BatchNorm2d(backend_output_channels))
        self._backend = torch.nn.Sequential(*mnasnet_layers)

        if not train_backend:
            for p in self._backend.parameters():
                p.requires_grad = False

        self._encoder = torch.nn.Sequential(
            torch.nn.Conv2d(backend_output_channels, backend_output_channels // 2, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(backend_output_channels // 2, backend_output_channels // 4, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(backend_output_channels // 4, backend_output_channels // 8, 1, stride=1, padding=0)
        )

        self._decoder = torch.nn.Sequential(
            torch.nn.Conv2d(backend_output_channels // 8, backend_output_channels // 4, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(backend_output_channels // 4, backend_output_channels // 2, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(backend_output_channels // 2, backend_output_channels, 1, stride=1, padding=0)
        )

    def forward(self, input):
        features = self._backend(input)
        features = F.interpolate(features, size=curious_dataset.LABEL_SIZE, mode='bilinear')

        x = self._encoder(features)
        x = self._decoder(x)

        error = x - features
        error = torch.pow(error, 2)
        error = torch.mean(error, dim=1)

        return error
