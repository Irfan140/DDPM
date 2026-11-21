"""The DDPM U-Net architecture used by training and inference."""

import math

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / (half - 1)
        )
        args = t[:, None] + freqs[None, :]
        return torch.cat((torch.sin(args), torch.cos(args)), dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(torch.relu(self.norm1(x)))
        h = h + self.time_proj(t)[:, :, None, None]
        h = self.conv2(torch.relu(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(batch, channels, height * width)
        k = self.k(h).reshape(batch, channels, height * width)
        v = self.v(h).reshape(batch, channels, height * width)
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(channels), dim=-1)
        h = (attn @ v.transpose(1, 2)).transpose(1, 2).reshape(
            batch, channels, height, width
        )
        return x + self.proj(h)


class DDPMUNet(nn.Module):
    def __init__(self, base_channels: int = 64):
        super().__init__()
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.conv_in = nn.Conv2d(1, base_channels, 3, padding=1)
        self.down1 = ResBlock(base_channels, base_channels, time_dim)
        self.down2 = ResBlock(base_channels, base_channels * 2, time_dim)
        self.pool = nn.AvgPool2d(2)
        self.mid1 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.attn = AttentionBlock(base_channels * 2)
        self.mid2 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.up1 = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        self.up2 = ResBlock(base_channels * 2, base_channels, time_dim)
        self.conv_out = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = self.time_mlp(t.float())
        x = self.conv_in(x)
        d1 = self.down1(x, t)
        d2 = self.down2(d1, t)
        h = self.pool(d2)
        h = self.mid1(h, t)
        h = self.attn(h)
        h = self.mid2(h, t)
        h = self.up(h)
        h = self.up1(h)
        h = torch.cat((h, d1), dim=1)
        h = self.up2(h, t)
        return self.conv_out(h)
