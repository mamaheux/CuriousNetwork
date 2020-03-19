import torch


class SmallCnnWithAutoencoder(torch.nn.Module):
    def __init__(self, kernel_size=3, first_output_channels=8, growth_rate=2):
        super(SmallCnnWithAutoencoder, self).__init__()
        if kernel_size % 2 == 0:
            raise ValueError('Invalid kernel size')

        conv_padding = kernel_size // 2
        self._cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3,
                            first_output_channels,
                            kernel_size, stride=1, padding=conv_padding),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(first_output_channels,
                            first_output_channels * growth_rate ** 1,
                            kernel_size, stride=1, padding=conv_padding),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(5, stride=5),
            torch.nn.Conv2d(first_output_channels * growth_rate ** 1,
                            first_output_channels * growth_rate ** 2,
                            kernel_size, stride=1, padding=conv_padding),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(first_output_channels * growth_rate ** 2,
                            first_output_channels * growth_rate ** 3,
                            kernel_size, stride=1, padding=conv_padding),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(3, stride=3),
            torch.nn.Conv2d(first_output_channels * growth_rate ** 3,
                            first_output_channels * growth_rate ** 4,
                            kernel_size, stride=1, padding=conv_padding),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(first_output_channels * growth_rate ** 4,
                            first_output_channels * growth_rate ** 5,
                            kernel_size, stride=1, padding=conv_padding),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.BatchNorm2d(first_output_channels * growth_rate ** 5)
        )

        cnn_output_channels = first_output_channels * growth_rate ** 5
        self._encoder = torch.nn.Sequential(
            torch.nn.Conv2d(cnn_output_channels, cnn_output_channels // 2, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(cnn_output_channels // 2, cnn_output_channels // 4, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(cnn_output_channels // 4, cnn_output_channels // 8, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(cnn_output_channels // 8, cnn_output_channels // 16, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
        )

        self._decoder = torch.nn.Sequential(
            torch.nn.Conv2d(cnn_output_channels // 16, cnn_output_channels // 8, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(cnn_output_channels // 8, cnn_output_channels // 4, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(cnn_output_channels // 4, cnn_output_channels // 2, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(cnn_output_channels // 2, cnn_output_channels, 1, stride=1, padding=0)
        )

    def forward(self, input):
        features = self._cnn(input)
        feature_count = features.size()[1]

        x = self._encoder(features)
        x = self._decoder(x)

        error = x - features
        error = torch.pow(error, 2)
        error = torch.sum(error, dim=1)
        error /= feature_count

        return error
