import torch


class SmallCnnWithAutoencoder(torch.nn.Module):
    def __init__(self):
        super(SmallCnnWithAutoencoder, self).__init__()
        self._cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(5, stride=5),
            torch.nn.Conv2d(6, 12, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(4, stride=4),
            torch.nn.Conv2d(12, 24, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(3, stride=3),
            torch.nn.Conv2d(24, 48, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.BatchNorm2d(48)
        )

        self._encoder = torch.nn.Sequential(
            torch.nn.Conv2d(48, 24, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(24, 12, 1, stride=1, padding=0),
            torch.nn.ReLU(True)
        )

        self._decoder = torch.nn.Sequential(
            torch.nn.Conv2d(12, 24, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(24, 48, 1, stride=1, padding=0)
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
