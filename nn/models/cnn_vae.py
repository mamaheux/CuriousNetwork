import torch
import torch.nn.functional as F

from models.base_model import BaseModel


# Receptive field of the encoder (if kernel_size=3) : 60x60
class CnnVae(BaseModel):
    def __init__(self, ini_feature_maps=4, feature_maps_growth_factor=2, kernel_size=3):
        super(CnnVae, self).__init__()

        if kernel_size % 2 == 0:
            raise ValueError('Kernel size must be an odd number')

        cnn_padding = kernel_size // 2;
        self._encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, ini_feature_maps, kernel_size, stride=1, padding=cnn_padding),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(ini_feature_maps,
                            ini_feature_maps * feature_maps_growth_factor ** 1,
                            kernel_size, stride=1, padding=cnn_padding),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(ini_feature_maps * feature_maps_growth_factor ** 1,
                            ini_feature_maps * feature_maps_growth_factor ** 2,
                            kernel_size, stride=1, padding=cnn_padding),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(ini_feature_maps * feature_maps_growth_factor ** 2,
                            ini_feature_maps * feature_maps_growth_factor ** 3,
                            kernel_size, stride=1, padding=cnn_padding),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(3, stride=3),
            torch.nn.Conv2d(ini_feature_maps * feature_maps_growth_factor ** 3,
                            ini_feature_maps * feature_maps_growth_factor ** 4,
                            kernel_size, stride=1, padding=cnn_padding),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(ini_feature_maps * feature_maps_growth_factor ** 4,
                            2 * ini_feature_maps * feature_maps_growth_factor ** 5,
                            kernel_size, stride=1, padding=cnn_padding),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(4, stride=4),
        )

        self._decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(ini_feature_maps * feature_maps_growth_factor ** 5,
                                     ini_feature_maps * feature_maps_growth_factor ** 4,
                                     kernel_size, stride=4, padding=0),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(ini_feature_maps * feature_maps_growth_factor ** 4,
                            ini_feature_maps * feature_maps_growth_factor ** 3,
                            kernel_size, stride=1, padding=cnn_padding),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ini_feature_maps * feature_maps_growth_factor ** 3,
                                     ini_feature_maps * feature_maps_growth_factor ** 2,
                                     kernel_size, stride=3, padding=cnn_padding),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(ini_feature_maps * feature_maps_growth_factor ** 2,
                            ini_feature_maps * feature_maps_growth_factor ** 1,
                            kernel_size, stride=1, padding=cnn_padding),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ini_feature_maps * feature_maps_growth_factor ** 1,
                                     ini_feature_maps,
                                     kernel_size, stride=2, padding=cnn_padding),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(ini_feature_maps, 3, kernel_size, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

        self._average_pool = torch.nn.AvgPool2d(30)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def internal_loss(self, use_gpu=False):
        loss = -0.5 * torch.sum(1 + self._logvar - self._mu.pow(2) - self._logvar.exp()) # KLD

        if torch.cuda.is_available() and use_gpu:
            loss = loss.cuda()

        return loss

    def forward(self, input):
        x = self._encoder(input)
        z_channels = x.size()[1] // 2
        self._mu = x[:, :z_channels, : ,:]
        self._logvar = x[:, z_channels:, :, :]

        z = self.reparameterize(self._mu, self._logvar)
        x = self._decoder(z)
        x = F.interpolate(x, size=(input.size()[2], input.size()[3]), mode='bilinear')

        error = x - input
        error = torch.pow(error, 2)
        error = self._average_pool(error)
        error = torch.mean(error, dim=1)

        return error
