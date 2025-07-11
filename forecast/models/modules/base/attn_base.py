import torch.nn as nn
import torch

from .others import Normalize
from einops import rearrange

from forecast.ops.flash_attention.flash_attention import FlashAttention

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.attention = FlashAttention(dim_head=64, dim=in_channels, heads=8)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):

        if len(x.shape) == 3:
            x = rearrange(x, 'b (h w) c -> b c h w', h=int(x.shape[1] ** 0.5))
        h_ = x
        h_ = self.norm(h_)

        h, w = x.shape[-2:]
        h_ = rearrange(h_, 'b c h w -> b (h w) c')
        h_ = self.attention(h_)
        h_ = rearrange(h_, 'b (h w) c -> b c h w', h=h, w=w)

        h_ = self.proj_out(h_)

        return x+h_