import torch
import torch.nn as nn


class Shrinkage_loss(nn.Module):
    def __init__(self, a, c):
        super(Shrinkage_loss, self).__init__()
        self._a = a
        self._c = c

    def forward(self, pred, gt):
        l1 = torch.mean((pred - gt))
        l2 = l1 ** 2
        shrinkage_loss = l2 / (1 + torch.exp(self._a * (self._c - l2)))
        return shrinkage_loss