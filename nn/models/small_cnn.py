import torch


class SmallCnnWithAutoencoder(torch.nn.Module):
    def __init__(self):
        super(SmallCnnWithAutoencoder, self).__init__()
        self._cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(8, 16, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(5, stride=5),
            torch.nn.Conv2d(16, 32, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(32, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(3, stride=3),
            torch.nn.Conv2d(64, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(128, 256, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.BatchNorm2d(256)
        )

        self._encoder = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(128, 64, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 32, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(32, 16, 1, stride=1, padding=0),
            torch.nn.ReLU(True)
        )

        self._decoder = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(32, 64, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 128, 1, stride=1, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(128, 256, 1, stride=1, padding=0)
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
