import torch.nn as nn
import torch

from einops import rearrange

def nonlinearity(x):
    return x * torch.sigmoid(x)

def Normalize(in_channels):
    if in_channels <= 32:
        num_groups = in_channels // 4
    else:
        num_groups = 32
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.in_channels = in_channels
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        bs = x.shape[0]
        x = torch.nn.functional.interpolate(x.to(torch.float32), scale_factor=2.0, mode="nearest").to(x.dtype)
        if self.with_conv:
            x = self.conv(x)
        x = rearrange(x, "b c h w -> b h w c", b=bs)
        return x