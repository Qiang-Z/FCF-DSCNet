import torch
from torch import nn
from torch.nn import functional as F


def norm(channels):
    groups = next(g for g in (8, 4, 2, 1) if channels % g == 0)
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), norm(cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), norm(cout), nn.SiLU(),
        )


class UpBlock(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.block = ConvBlock(cin + skip, cout)

    def forward(self, x, skip):
        return self.block(torch.cat([
            F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip
        ], dim=1))


def patchify(x, patch):
    """B,C,H,W -> B,Nh,Nw,P,C; pad on bottom/right only."""
    b, c, h, w = x.shape
    ph, pw = (-h) % patch, (-w) % patch
    x = F.pad(x, (0, pw, 0, ph))
    nh, nw = x.shape[-2] // patch, x.shape[-1] // patch
    x = x.reshape(b, c, nh, patch, nw, patch).permute(0, 2, 4, 3, 5, 1)
    return x.reshape(b, nh, nw, patch * patch, c), (h, w)


def unpatchify(x, patch, size):
    b, nh, nw, _, c = x.shape
    x = x.reshape(b, nh, nw, patch, patch, c).permute(0, 5, 1, 3, 2, 4)
    return x.reshape(b, c, nh * patch, nw * patch)[..., :size[0], :size[1]]


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp_min(1)

