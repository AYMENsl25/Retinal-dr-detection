"""Vessel segmentation model from `transunet-vessels-segmentation.ipynb`.

The notebook trained a TransUNet-style architecture:
ResNet50 encoder -> transformer blocks -> CNN decoder.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


def _resnet50(pretrained: bool) -> nn.Module:
    if not pretrained:
        return models.resnet50(weights=None)
    try:
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    except AttributeError:
        return models.resnet50(pretrained=True)


class TransUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        img_size: int = 512,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        pretrained: bool = False,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("The notebook TransUNet was trained with 3-channel RGB input.")

        self.img_size = img_size
        resnet = _resnet50(pretrained)

        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3

        self.proj = nn.Conv2d(1024, embed_dim, 1)
        self.feat_size = img_size // 16
        num_patches = self.feat_size**2

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.dec4 = DecoderBlock(embed_dim, 512, 256)
        self.dec3 = DecoderBlock(256, 256, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 0, 32)

        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        t = self.proj(s4)
        batch_size, channels, height, width = t.shape
        t = t.flatten(2).transpose(1, 2)
        t = t + self.pos_embed[:, : t.size(1), :]
        t = self.transformer(t)
        t = self.norm(t)
        t = t.transpose(1, 2).view(batch_size, channels, height, width)

        d4 = self.dec4(t, s3)
        d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2)

        out = F.interpolate(d1, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        return self.final(out)


def build_vessel_model(img_size: int = 512, pretrained: bool = False) -> TransUNet:
    return TransUNet(
        in_channels=3,
        out_channels=1,
        img_size=img_size,
        embed_dim=768,
        num_heads=12,
        num_layers=6,
        pretrained=pretrained,
    )
