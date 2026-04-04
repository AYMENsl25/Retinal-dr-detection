"""
Retinal DR Detection — Lesion Training Utilities
==================================================
Multi-label loss function and training helpers for 5-channel
lesion segmentation.

IMPORTANT difference from vessel segmentation:
  - Vessel: single channel  → loss computed on flat (B*H*W) tensor
  - Lesion: five channels    → loss computed PER CHANNEL then averaged

This ensures the model learns each lesion type equally, even if some
types (like MA) cover very few pixels compared to others (like OD).
"""

import torch
import torch.nn as nn


class MultiLabelDiceBCELoss(nn.Module):
    """
    Combined Dice + BCE loss for multi-label segmentation.

    How it works:
      1. For EACH of the 5 lesion channels independently:
         - Compute Dice Loss  (directly optimises our evaluation metric)
         - Compute BCE Loss   (provides stable per-pixel gradients)
         - Combine:  channel_loss = dice_w * Dice + bce_w * BCE
      2. Average across all channels → final scalar loss

    Why per-channel?
      If we flattened all 5 channels together, the loss would be
      dominated by whatever lesion type covers the most pixels (e.g. OD).
      Per-channel averaging gives each lesion type equal importance.

    Args:
        dice_weight : Weight for the Dice component (default 0.5)
        bce_weight  : Weight for the BCE component  (default 0.5)
        smooth      : Smoothing factor to avoid division by zero
    """

    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def _dice_loss_per_channel(self, pred_logits, targets):
        """Compute Dice loss for each channel separately."""
        pred = torch.sigmoid(pred_logits)               # (B, C, H, W)
        B, C, H, W = pred.shape

        # Reshape to (C, B*H*W) so each channel is independent
        pred_flat = pred.permute(1, 0, 2, 3).reshape(C, -1)
        tgt_flat  = targets.permute(1, 0, 2, 3).reshape(C, -1)

        intersection = (pred_flat * tgt_flat).sum(dim=1)      # (C,)
        union = pred_flat.sum(dim=1) + tgt_flat.sum(dim=1)    # (C,)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()   # Average over channels

    def _bce_loss_per_channel(self, pred_logits, targets):
        """Compute BCE loss averaged over channels."""
        # BCEWithLogitsLoss with reduction='none' gives (B, C, H, W)
        bce_all = self.bce(pred_logits, targets)
        # Average over pixels (H, W), then over batch (B), then channels (C)
        return bce_all.mean()

    def forward(self, pred_logits, targets):
        """
        Args:
            pred_logits : (B, C, H, W) raw model output BEFORE sigmoid
            targets     : (B, C, H, W) binary ground truth
        Returns:
            Scalar loss value
        """
        dice = self._dice_loss_per_channel(pred_logits, targets)
        bce  = self._bce_loss_per_channel(pred_logits, targets)
        return self.dice_weight * dice + self.bce_weight * bce
