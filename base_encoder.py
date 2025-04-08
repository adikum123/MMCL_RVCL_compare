import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    def __init__(self, ckpt, device):
        super(BaseEncoder, self).__init__()
        self.encoder = torch.load(ckpt, device)

    def forward(self, x):
        return self.encoder(x)

    def set_eval(self):
        self.encoder.eval()