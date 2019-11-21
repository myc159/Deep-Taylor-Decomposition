from .resnet import *
import torch
import torch.nn as nn

class Forward(nn.Module):
    def __init__(self,):
        super(Forward, self).__init__()

        self.model = resnet50()

    def forward(self, x):
        return self.model(x)
