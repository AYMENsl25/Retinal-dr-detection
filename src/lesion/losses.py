"""
Retinal DR Detection — Lesion Training Utilities
==================================================
Multi-label loss function and training helpers for 5-channel
lesion segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelDiceBCELoss(nn.Module):
    """
    Combined Dice + Focal Loss (previously BCE) for multi-label segmentation.
    (Kept class name identical so old notebooks don't break on import)

    How it works:
      1. For EACH of the 5 lesion channels independently:
         - Compute Dice Loss  (directly optimises our evaluation metric)
         - Compute FOCAL Loss (punishes the model 100x harder for missing tiny details like MA)
         - Combine:  channel_loss = dice_w * Dice + focal_w * Focal
      2. Average across all channels → final scalar loss
    
    Args:
        dice_weight : Weight for the Dice component (default 0.8)
        bce_weight  : Weight for the Focal component (default 0.2)
        smooth      : Smoothing factor to avoid division by zero
    """

    def __init__(self, dice_weight=0.8, bce_weight=0.2, smooth=1.0, alpha=0.5, gamma=2.0):
        super().__init__()
        # We shift to 80% Dice, 20% Focal because Dice is better for class imbalance.
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma
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

        # Using smooth=1.0 prevents exploding gradients when the mask is completely empty (Black)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()   # Average over channels

    def _focal_loss_per_channel(self, pred_logits, targets):
        """Compute Focal Loss averaged over channels."""
        # 1. Compute standard BCE per pixel
        bce_loss = self.bce(pred_logits, targets)
        
        # 2. Convert standard BCE to probabilities
        pt = torch.exp(-bce_loss)
        
        # 3. Apply Focal Loss formula: decreases loss for easy pixels (the giant black background)
        #    and increases loss for hard pixels (the tiny Microaneurysms)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        # Average over batch and spatial dims
        return focal_loss.mean()

    def forward(self, pred_logits, targets):
        """
        Args:
            pred_logits : (B, C, H, W) raw model output BEFORE sigmoid
            targets     : (B, C, H, W) binary ground truth
        Returns:
            Scalar loss value
        """
        dice = self._dice_loss_per_channel(pred_logits, targets)
        focal = self._focal_loss_per_channel(pred_logits, targets)
        return self.dice_weight * dice + self.bce_weight * focal
