import torch

class CnnAutoencoder(torch.nn.Module):
    def __init__(self):
        super(CnnAutoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(16, 8, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(8, 16, 3, stride=2, padding=0),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(8, 3, 2, stride=1, padding=0),
            torch.nn.Sigmoid()
        )
        
        self.average_pool = torch.nn.AvgPool2d(120)

    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)
        error = x - input
        error = torch.pow(error, 2)
        error = self.average_pool(error)
        error = torch.sum(error, dim=1)
        return error
