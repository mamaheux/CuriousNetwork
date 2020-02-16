import torch

class CnnAutoencoder(torch.nn.Module):
    def __init__(self):
        super(CnnAutoencoder, self).__init__()
        self._encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(16, 8, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(8, 1, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2)
        )

        self._decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1, 8, 3, stride=2, padding=0),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(8, 3, 2, stride=1, padding=0),
            torch.nn.Sigmoid()
        )
        
        self._average_pool = torch.nn.AvgPool2d(120)

    def forward(self, input):
        input_channel_count = input.size()[1]

        x = self._encoder(input)
        x = self._decoder(x)

        error = x - input
        error = torch.pow(error, 2)
        error = self._average_pool(error)
        error = torch.sum(error, dim=1)
        error /= input_channel_count

        return error
