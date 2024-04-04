import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

import einops
from einops import rearrange
import numpy as np
import pytorch_colors as colors

class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        global embedding, norm
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) +
                         self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / \
                (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / \
                (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate

class Diffusion(nn.Module):
    def __init__(self, inp, dim, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(Diffusion, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio)

        self.gate = GCT(dim)
        self.gate_2 = GCT(init_channels)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(), )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),)

    def forward(self, x):
        m_batchsize, N, C, height, width = x.size()

        x_input = x.view(m_batchsize, N * C, height, width)
        x_input = self.gate(x_input)
        # print(x_input.shape)
        x1 = self.primary_conv(x_input)  # 主要的卷积操作
        x1 = self.gate_2(x1)
        x2 = self.cheap_operation(x1)  # cheap变换操作
        out = torch.cat([x1, x2], dim=1)  # 二者cat到一起
        out = out[:, :self.oup, :, :]
        # print(out.shape)
        return out


diffusion = Diffusion(48, 48, 48)

x = torch.randn(1,3,16,32,32)

# x1 = diffusion(x)

print(x)