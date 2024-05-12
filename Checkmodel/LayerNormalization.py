import torch.nn as nn
import torch

class LayerNormalization(nn.Module):
    """ 批量层归一化 """
    def __init__(self, d_model, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True,)
        std = z.std(dim=-1, keepdim=True,)
        ln_out = (z - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta
        return ln_out