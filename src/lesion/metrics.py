"""
Retinal DR Detection — Lesion Evaluation Metrics
==================================================
Per-channel and overall metrics for multi-label lesion segmentation.

For each of the 5 lesion types we compute:
  - Dice Coefficient  (primary metric)
  - IoU / Jaccard Index
  - Sensitivity (Recall)  — "did we find all lesion pixels?"
  - Specificity            — "did we avoid false alarms?"
  - Precision (PPV)

Then we report both per-type scores AND macro-averaged scores.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from src.lesion.dataset import LESION_TYPES


def evaluate_multilabel_batch(pred_logits, targets, threshold=0.5):
    """
    Compute metrics for a batch of multi-label predictions.

    Args:
        pred_logits : (B, C, H, W)  raw model output BEFORE sigmoid
        targets     : (B, C, H, W)  binary ground truth
        threshold   : Binarization threshold for sigmoid output

    Returns:
        dict with per-channel and mean metrics
        Example: {'dice_MA': 0.72, ..., 'dice_mean': 0.68, ...}
    """
    smooth = 1e-6
    with torch.no_grad():
        pred_probs  = torch.sigmoid(pred_logits)
        pred_binary = (pred_probs > threshold).float()

        B, C, H, W = pred_binary.shape
        metrics = {}

        for ch in range(C):
            p = pred_binary[:, ch, :, :].contiguous().view(-1)
            t = targets[:, ch, :, :].contiguous().view(-1)

            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()
            tn = ((1 - p) * (1 - t)).sum()

            dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
            iou  = (tp + smooth) / (tp + fp + fn + smooth)
            sens = (tp + smooth) / (tp + fn + smooth)
            spec = (tn + smooth) / (tn + fp + smooth)
            prec = (tp + smooth) / (tp + fp + smooth)

            lt = LESION_TYPES[ch] if ch < len(LESION_TYPES) else f'ch{ch}'
            metrics[f'dice_{lt}'] = dice.item()
            metrics[f'iou_{lt}']  = iou.item()
            metrics[f'sens_{lt}'] = sens.item()
            metrics[f'spec_{lt}'] = spec.item()
            metrics[f'prec_{lt}'] = prec.item()

        # Macro averages
        for m in ['dice', 'iou', 'sens', 'spec', 'prec']:
            vals = [metrics[f'{m}_{LESION_TYPES[c]}'] for c in range(C)]
            metrics[f'{m}_mean'] = np.mean(vals)

    return metrics


def evaluate_multilabel_full(pred_probs_all, targets_all,
                             num_classes=None, threshold=0.5):
    """
    Compute ALL metrics on the full test set (after collecting
    all predictions across batches).

    Args:
        pred_probs_all : np.array (N, C, H, W)  sigmoid probabilities
        targets_all    : np.array (N, C, H, W)  binary ground truth
        num_classes    : Number of channels (default: len(LESION_TYPES))

    Returns:
        dict with per-channel AND macro-averaged metrics including AUC
    """
    smooth = 1e-6
    if num_classes is None:
        num_classes = len(LESION_TYPES)

    pred_binary = (pred_probs_all > threshold).astype(np.float32)
    results = {}

    for ch in range(num_classes):
        p_flat = pred_binary[:, ch, :, :].ravel()
        t_flat = targets_all[:, ch, :, :].ravel()
        prob_flat = pred_probs_all[:, ch, :, :].ravel()

        tp = (p_flat * t_flat).sum()
        fp = (p_flat * (1 - t_flat)).sum()
        fn = ((1 - p_flat) * t_flat).sum()
        tn = ((1 - p_flat) * (1 - t_flat)).sum()

        dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        iou  = (tp + smooth) / (tp + fp + fn + smooth)
        sens = (tp + smooth) / (tp + fn + smooth)
        spec = (tn + smooth) / (tn + fp + smooth)
        prec = (tp + smooth) / (tp + fp + smooth)
        acc  = (tp + tn) / (tp + tn + fp + fn + smooth)

        # AUC metrics (may fail if all labels are the same)
        try:
            auc_roc = roc_auc_score(t_flat, prob_flat)
        except ValueError:
            auc_roc = 0.0
        try:
            auc_pr = average_precision_score(t_flat, prob_flat)
        except ValueError:
            auc_pr = 0.0

        lt = LESION_TYPES[ch] if ch < len(LESION_TYPES) else f'ch{ch}'
        results[f'dice_{lt}']    = float(dice)
        results[f'iou_{lt}']     = float(iou)
        results[f'sens_{lt}']    = float(sens)
        results[f'spec_{lt}']    = float(spec)
        results[f'prec_{lt}']    = float(prec)
        results[f'acc_{lt}']     = float(acc)
        results[f'auc_roc_{lt}'] = float(auc_roc)
        results[f'auc_pr_{lt}']  = float(auc_pr)

    # Macro averages
    for m in ['dice', 'iou', 'sens', 'spec', 'prec', 'acc',
              'auc_roc', 'auc_pr']:
        vals = [results[f'{m}_{LESION_TYPES[c]}'] for c in range(num_classes)]
        results[f'{m}_mean'] = float(np.mean(vals))

    return results
