import torch

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def internal_loss(self, use_gpu=False):
        if torch.cuda.is_available() and use_gpu:
            return torch.zeros(1).cuda()

        return torch.zeros(1)
