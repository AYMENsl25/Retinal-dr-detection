"""Vessel segmentation model architectures.

Three models are ensembled for vessel segmentation:
  1. CSNet  — U-Net with Channel-Spatial Attention
  2. AttentionUNet — smp.Unet with SCSE attention (resnet34 encoder)
  3. SwinUNet — Swin Transformer encoder + U-Net decoder

Ensemble: 0.4 * CSNet + 0.4 * AttentionUNet + 0.2 * SwinUNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CSNet — Channel-Spatial Attention U-Net
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """Spatial attention using max+avg pooled features."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(combined)
        return x * attn


class CSBlock(nn.Module):
    """Channel-Spatial Attention Block."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CSNet(nn.Module):
    """CS-Net: U-Net with Channel-Spatial Attention at every encoder and skip level."""

    def __init__(self, in_channels=3, out_channels=1, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        # Encoder
        self.encoders = nn.ModuleList()
        self.cs_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch = in_channels
        for f in features:
            self.encoders.append(ConvBlock(ch, f))
            self.cs_blocks.append(CSBlock(f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        self.cs_bottleneck = CSBlock(features[-1] * 2)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        rev = list(reversed(features))
        prev = features[-1] * 2
        for f in rev:
            self.upconvs.append(nn.ConvTranspose2d(prev, f, 2, stride=2))
            self.decoders.append(ConvBlock(f * 2, f))
            prev = f

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for enc, cs, pool in zip(self.encoders, self.cs_blocks, self.pools):
            x = enc(x)
            x = cs(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        x = self.cs_bottleneck(x)

        skips = skips[::-1]
        for up, dec, skip in zip(self.upconvs, self.decoders, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.final(x)


# ---------------------------------------------------------------------------
# SwinUNet — Swin Transformer encoder + U-Net decoder
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """Decoder block: upsample + concat skip + convolutions."""

    def __init__(self, in_ch, skip_ch, out_ch):
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

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SwinUNet(nn.Module):
    """Swin Transformer encoder with U-Net style decoder."""

    def __init__(self, out_channels=1, pretrained=False):
        super().__init__()
        import timm

        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            img_size=512,
        )

        self.encoder_channels = self.encoder.feature_info.channels()

        # Decoder
        self.dec4 = DecoderBlock(self.encoder_channels[3], self.encoder_channels[2], 256)
        self.dec3 = DecoderBlock(256, self.encoder_channels[1], 128)
        self.dec2 = DecoderBlock(128, self.encoder_channels[0], 64)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(32, out_channels, 1)

    def _fix_feature_format(self, features):
        """Fix timm Swin outputs from (B,H,W,C) to (B,C,H,W)."""
        fixed = []
        for feat, expected_c in zip(features, self.encoder_channels):
            if feat.shape[1] != expected_c:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            fixed.append(feat)
        return fixed

    def forward(self, x):
        features = self.encoder(x)
        features = self._fix_feature_format(features)

        d4 = self.dec4(features[3], features[2])
        d3 = self.dec3(d4, features[1])
        d2 = self.dec2(d3, features[0])

        out = self.final_up(d2)
        out = F.interpolate(out, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        return self.final_conv(out)
