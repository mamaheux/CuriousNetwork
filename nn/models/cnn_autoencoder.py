import torch

class CnnAutoencoder(torch.nn.Module):
    def __init__(self):
        super(CnnAutoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 20, 3, stride=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(20, 20, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(20, 20, 3, stride=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(20, 20, 3, stride=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(20, 3, 3, stride=3, padding=1),
            torch.nn.Sigmoid()
        )
        
        self.average_pool = torch.nn.AvgPool2d(120)

    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)
        error = x - input
        error = torch.sqrt(torch.mul(error, error))
        error = self.average_pool(error)
        return error
