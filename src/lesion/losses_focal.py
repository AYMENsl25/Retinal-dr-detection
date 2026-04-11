"""
Retinal DR Detection — Lesion Training Utilities (V2)
==================================================
Multi-label loss function using FOCAL LOSS.

Key insight: Advanced networks (MultiRes, BiDLKA) learn to "cheat" BCE Loss 
by predicting 100% background on tiny lesions (MA, HE) because they are so small.
This script implements Sigmoid Focal Loss, heavily punishing False Negatives 
(missed lesions) by down-weighting the vast amounts of easy background pixels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Binary Focal Loss designed to address extreme class imbalance.
    
    Formula: -alpha * (1 - pt)^gamma * log(pt) - (1-alpha) * pt^gamma * log(1-pt)
    
    gamma: Focusing parameter. Higher gamma heavily penalizes hard misclassifications
           while completely ignoring easy background pixels.
    alpha: Balance weight. Alpha > 0.5 weights the positive class (lesions) higher.
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are raw logits before sigmoid
        p = torch.sigmoid(inputs)
        
        # Calculate standard BCE
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        
        # pt is the predicted probability for the TRUE class
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # focal modulating factor: (1 - pt)^gamma
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            # apply alpha weighting: alpha for positive class, (1-alpha) for negative
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
            
        return loss


class MultiLabelDiceFocalLoss(nn.Module):
    """
    Weighted Dice + Focal Loss for multi-label segmentation.
    This replaces MultiLabelDiceBCELoss to solve the "Tiny Lesion" problem in advanced models.

    Args:
        dice_weight    : Weight for the Dice component (default 0.5)
        focal_weight   : Weight for the Focal component (default 0.5)
        alpha          : Focal Loss Foreground Weight (default 0.75 pushes harder on Foregrounds)
        gamma          : Focal Loss Modulation (default 2.0 ignores easy background entirely)
        smooth         : Smoothing factor for Dice (default 1e-6)
        channel_weights: Per-channel multiplier [MA, HE, EX, OD, CW]
    """

    def __init__(self, dice_weight=0.5, focal_weight=0.5, alpha=0.75, gamma=2.0, smooth=1e-6,
                 channel_weights=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth
        
        # Initialize custom Focal Loss instead of BCE
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction='none')

        # Penalize ignoring tiny lesions (same as V1)
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

    def _focal_loss_per_channel(self, pred_logits, targets):
        """Compute WEIGHTED Focal loss per channel."""
        # focal_all shape: (B, C, H, W)
        focal_all = self.focal(pred_logits, targets)

        # Average over batch and spatial dims → per-channel loss (C,)
        per_channel = focal_all.mean(dim=(0, 2, 3))  # (C,)

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
        focal = self._focal_loss_per_channel(pred_logits, targets)
        return self.dice_weight * dice + self.focal_weight * focal
