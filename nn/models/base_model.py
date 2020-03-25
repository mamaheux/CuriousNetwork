import torch

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def internal_loss(self):
        return torch.zeros(1)
