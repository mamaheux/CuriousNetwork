import torch

from models.base_model import BaseModel
from models.cnn_blocks import DenseBlock


class SmallCnnWithAutoencoderDenseBlocks(BaseModel):
    def __init__(self, kernel_size=3, growth_rate=2, useBatchNorm=False):
        super(SmallCnnWithAutoencoderDenseBlocks, self).__init__()
        if kernel_size % 2 == 0:
            raise ValueError('Invalid kernel size')

        self._cnn = torch.nn.Sequential(
            DenseBlock(2, 3, growth_rate),

            DenseBlock(2, 3 + growth_rate * 2, growth_rate),
            torch.nn.MaxPool2d(5, stride=5),

            DenseBlock(2, 3 + growth_rate * 4, growth_rate),

            DenseBlock(2, 3 + growth_rate * 6, growth_rate),
            torch.nn.MaxPool2d(3, stride=3),

            DenseBlock(2, 3 + growth_rate * 8, growth_rate),

            DenseBlock(2, 3 + growth_rate * 10, growth_rate),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.BatchNorm2d(3 + growth_rate * 12)
        )

        cnn_output_channels = 3 + growth_rate * 12
        self._encoder = torch.nn.Sequential(
            torch.nn.Conv2d(cnn_output_channels, cnn_output_channels // 2, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(cnn_output_channels // 2) if useBatchNorm else torch.nn.Sequential(),
            torch.nn.Conv2d(cnn_output_channels // 2, cnn_output_channels // 4, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(cnn_output_channels // 4) if useBatchNorm else torch.nn.Sequential(),
            torch.nn.Conv2d(cnn_output_channels // 4, cnn_output_channels // 8, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(cnn_output_channels // 8) if useBatchNorm else torch.nn.Sequential(),
            torch.nn.Conv2d(cnn_output_channels // 8, cnn_output_channels // 16, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(cnn_output_channels // 16) if useBatchNorm else torch.nn.Sequential(),
        )

        self._decoder = torch.nn.Sequential(
            torch.nn.Conv2d(cnn_output_channels // 16, cnn_output_channels // 8, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(cnn_output_channels // 8) if useBatchNorm else torch.nn.Sequential(),
            torch.nn.Conv2d(cnn_output_channels // 8, cnn_output_channels // 4, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(cnn_output_channels // 4) if useBatchNorm else torch.nn.Sequential(),
            torch.nn.Conv2d(cnn_output_channels // 4, cnn_output_channels // 2, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(cnn_output_channels // 2) if useBatchNorm else torch.nn.Sequential(),
            torch.nn.Conv2d(cnn_output_channels // 2, cnn_output_channels, 1, stride=1, padding=0)
        )

    def forward(self, input):
        features = self._cnn(input)

        x = self._encoder(features)
        x = self._decoder(x)

        error = x - features
        error = torch.pow(error, 2)
        error = torch.mean(error, dim=1)

        return error
