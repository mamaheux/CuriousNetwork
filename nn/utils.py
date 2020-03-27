import torch

class MinMaxNormalization:
    def __call__(self, x):
        min_value = x.min()
        range_value = x.max() - min_value
        if range_value > 0:
            return (x - min_value) / range_value
        else:
            return torch.zeros(x.size())
