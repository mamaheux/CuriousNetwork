import torch
import torch.nn.functional as F


class CnnAutoencoder(torch.nn.Module):
    def __init__(self):
        super(CnnAutoencoder, self).__init__()
        self._encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(4, 8, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(5, stride=5),
            torch.nn.Conv2d(8, 16, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(16, 32, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(3, stride=3),
            torch.nn.Conv2d(32, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),
        )

        self._decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 32, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(32, 16, 3, stride=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(16, 8, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(8, 4, 3, stride=5, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(4, 3, 3, stride=1, padding=0),
            torch.nn.Sigmoid()
        )
        
        self._average_pool = torch.nn.AvgPool2d(30)

    def forward(self, input):
        x = self._encoder(input)
        x = self._decoder(x)
        x = F.interpolate(x, size=(input.size()[2], input.size()[3]), mode='bilinear')

        error = x - input
        error = torch.pow(error, 2)
        error = self._average_pool(error)
        error = torch.mean(error, dim=1)

        return error
