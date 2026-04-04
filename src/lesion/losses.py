"""
Retinal DR Detection — Lesion Training Utilities
==================================================
Multi-label loss function for 5-channel lesion segmentation.

Key insight: OD is huge (~5% of pixels), MA is tiny (~0.003% of pixels).
If we weight all channels equally, the model just learns OD and ignores
everything else. Solution: give HIGHER weight to rare/tiny lesion types.
"""

import torch
import torch.nn as nn


class MultiLabelDiceBCELoss(nn.Module):
    """
    Weighted Dice + BCE loss for multi-label segmentation.

    WHAT CHANGED from the original version:
      - Each lesion channel now has its own weight in the loss
      - Tiny lesions (MA, HE, CW) get HIGHER weights
      - Large lesions (OD) get LOWER weights
      - This forces the model to spend more effort learning tiny structures

    The weights are based on how rare each lesion type is:
      MA  = 3.0  (tiny dots, hardest to detect → highest weight)
      HE  = 2.5  (small blobs, hard)
      EX  = 1.5  (medium patches)
      OD  = 0.5  (giant circle, easy → lowest weight)
      CW  = 2.5  (small patches, hard)

    Args:
        dice_weight    : Weight for the Dice component (default 0.5)
        bce_weight     : Weight for the BCE component  (default 0.5)
        smooth         : Smoothing factor to avoid division by zero
        channel_weights: Per-channel importance weights [MA, HE, EX, OD, CW]
    """

    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6,
                 channel_weights=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        # Default weights: penalise ignoring tiny lesions
        if channel_weights is None:
            #                  MA   HE   EX   OD   CW
            channel_weights = [3.0, 2.5, 1.5, 0.5, 2.5]
        self.register_buffer(
            'channel_weights',
            torch.tensor(channel_weights, dtype=torch.float32)
        )

    def _dice_loss_per_channel(self, pred_logits, targets):
        """Compute WEIGHTED Dice loss for each channel separately."""
        pred = torch.sigmoid(pred_logits)               # (B, C, H, W)
        B, C, H, W = pred.shape

        # Reshape to (C, B*H*W) so each channel is independent
        pred_flat = pred.permute(1, 0, 2, 3).reshape(C, -1)
        tgt_flat  = targets.permute(1, 0, 2, 3).reshape(C, -1)

        intersection = (pred_flat * tgt_flat).sum(dim=1)      # (C,)
        union = pred_flat.sum(dim=1) + tgt_flat.sum(dim=1)    # (C,)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice   # (C,)

        # Apply channel weights: MA loss × 3.0, OD loss × 0.5, etc.
        w = self.channel_weights.to(dice_loss.device)
        weighted = (dice_loss * w).sum() / w.sum()
        return weighted

    def _bce_loss_per_channel(self, pred_logits, targets):
        """Compute WEIGHTED BCE loss per channel."""
        B, C, H, W = pred_logits.shape
        bce_all = self.bce(pred_logits, targets)  # (B, C, H, W)

        # Average over batch and spatial dims → per-channel loss (C,)
        per_channel = bce_all.mean(dim=(0, 2, 3))  # (C,)

        # Apply channel weights
        w = self.channel_weights.to(per_channel.device)
        weighted = (per_channel * w).sum() / w.sum()
        return weighted

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
