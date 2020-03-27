import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, nb_layers,  in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.bns = []
        self.convs = []
        for i in range(nb_layers):
            kernel_size = 1 if i % 2 == 0 else 3
            padding = 0 if i % 2 == 0 else 1

            self.bns.append(nn.BatchNorm2d(in_channels + i * growth_rate))
            self.convs.append(nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=kernel_size,
                                        padding=padding))

            self.add_module('bn_' + str(i), self.bns[i])
            self.add_module('conv_' + str(i), self.convs[i])

    def forward(self, x):
        output = x

        for i in range(len(self.convs)):
            input = output
            output = self.convs[i](F.relu(self.bns[i](input)))
            output = torch.cat([input, output], 1)

        return output
