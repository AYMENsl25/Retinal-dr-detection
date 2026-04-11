"""
Retinal DR Detection - Medical Evaluation Metrics
===================================================
Comprehensive metrics for evaluating segmentation quality in clinical context.

In medical imaging, standard accuracy is NOT enough. A model that predicts
"no vessel everywhere" would get ~90% accuracy (because vessels are only ~10%
of the image), but would be completely useless clinically.

Key metrics explained:
- Dice Coefficient: Overlap between prediction and ground truth (harmonic mean of P & R)
- IoU (Jaccard): Intersection over Union — stricter than Dice
- Sensitivity (Recall): "Of all actual vessels, how many did we detect?"
  → Critical in medicine: missing a vessel = missing pathology
- Specificity: "Of all non-vessel pixels, how many did we correctly identify?"
  → Important to avoid false alarms
- Precision (PPV): "Of everything we predicted as vessel, how much was correct?"
- AUC-ROC: Overall discriminative ability across all thresholds
- AUC-PR: Better than AUC-ROC for imbalanced data (which vessel segmentation is)
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


# ============================================================
# Pixel-Level Metrics (operate on binary masks)
# ============================================================

def dice_coefficient(pred, target, smooth=1e-6):
    """
    Dice = 2 * |P ∩ T| / (|P| + |T|)
    
    Ranges from 0 (no overlap) to 1 (perfect overlap).
    This is THE standard metric for medical image segmentation.
    
    Args:
        pred: Binary prediction tensor (B, 1, H, W)
        target: Binary ground truth tensor (B, 1, H, W)
        smooth: Smoothing factor to avoid division by zero
    """
    pred_flat = pred.contiguous().view(-1).float()
    target_flat = target.contiguous().view(-1).float()
    
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def iou_score(pred, target, smooth=1e-6):
    """
    IoU (Jaccard Index) = |P ∩ T| / |P ∪ T|
    
    Stricter than Dice — penalizes false positives and false negatives more.
    IoU of 0.5 means the prediction overlaps only half the ground truth area.
    """
    pred_flat = pred.contiguous().view(-1).float()
    target_flat = target.contiguous().view(-1).float()
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def sensitivity(pred, target, smooth=1e-6):
    """
    Sensitivity (Recall / True Positive Rate)
    = TP / (TP + FN)
    
    "Of all real vessels in the image, what fraction did our model find?"
    In clinical settings this is CRITICAL — a missed vessel could mean
    missing a hemorrhage or microaneurysm.
    """
    pred_flat = pred.contiguous().view(-1).float()
    target_flat = target.contiguous().view(-1).float()
    
    tp = (pred_flat * target_flat).sum()
    fn = (target_flat * (1 - pred_flat)).sum()
    return (tp + smooth) / (tp + fn + smooth)


def specificity(pred, target, smooth=1e-6):
    """
    Specificity (True Negative Rate)
    = TN / (TN + FP)
    
    "Of all background pixels, how many did we correctly leave as background?"
    High specificity means fewer false alarms.
    """
    pred_flat = pred.contiguous().view(-1).float()
    target_flat = target.contiguous().view(-1).float()
    
    tn = ((1 - pred_flat) * (1 - target_flat)).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    return (tn + smooth) / (tn + fp + smooth)


def precision_score(pred, target, smooth=1e-6):
    """
    Precision (Positive Predictive Value)
    = TP / (TP + FP)
    
    "Of everything our model labeled as vessel, how much actually was vessel?"
    """
    pred_flat = pred.contiguous().view(-1).float()
    target_flat = target.contiguous().view(-1).float()
    
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    return (tp + smooth) / (tp + fp + smooth)


def pixel_accuracy(pred, target):
    """
    Standard pixel-wise accuracy. Included for completeness but NOT
    reliable for segmentation due to class imbalance.
    """
    pred_flat = pred.contiguous().view(-1).float()
    target_flat = target.contiguous().view(-1).float()
    correct = (pred_flat == target_flat).sum()
    total = target_flat.numel()
    return correct.float() / total


# ============================================================
# Threshold-Independent Metrics (operate on probability maps)
# ============================================================

def compute_auc_roc(pred_probs, targets):
    """
    AUC-ROC: Area Under the Receiver Operating Characteristic Curve.
    
    Measures overall discriminative ability across ALL possible thresholds.
    A value of 0.5 means random guessing; 1.0 means perfect separation.
    
    Args:
        pred_probs: Raw sigmoid probabilities (numpy, flattened)
        targets: Binary ground truth (numpy, flattened)
    """
    try:
        return roc_auc_score(targets, pred_probs)
    except ValueError:
        return 0.0


def compute_auc_pr(pred_probs, targets):
    """
    AUC-PR: Area Under the Precision-Recall Curve.
    
    More informative than AUC-ROC for imbalanced datasets (which vessel
    segmentation always is — vessels are ~10% of pixels).
    """
    try:
        return average_precision_score(targets, pred_probs)
    except ValueError:
        return 0.0


# ============================================================
# Batch Evaluation Helper
# ============================================================

def evaluate_batch(pred_logits, targets, threshold=0.5):
    """
    Compute ALL metrics for a single batch.
    
    Args:
        pred_logits: Raw model output BEFORE sigmoid (B, 1, H, W)
        targets: Binary ground truth (B, 1, H, W)
        threshold: Binarization threshold for the sigmoid output
        
    Returns:
        dict with all metric values
    """
    with torch.no_grad():
        pred_probs = torch.sigmoid(pred_logits)
        pred_binary = (pred_probs > threshold).float()
        
        metrics = {
            'dice': dice_coefficient(pred_binary, targets).item(),
            'iou': iou_score(pred_binary, targets).item(),
            'sensitivity': sensitivity(pred_binary, targets).item(),
            'specificity': specificity(pred_binary, targets).item(),
            'precision': precision_score(pred_binary, targets).item(),
            'accuracy': pixel_accuracy(pred_binary, targets).item(),
        }
        
    return metrics


def evaluate_full(pred_probs_all, targets_all, threshold=0.5):
    """
    Compute ALL metrics including AUC on the full dataset (after collecting
    all predictions across batches).
    
    Args:
        pred_probs_all: numpy array of all sigmoid probabilities (flattened)
        targets_all: numpy array of all binary targets (flattened)
    """
    pred_binary = (pred_probs_all > threshold).astype(np.float32)
    
    # Pixel-level metrics
    tp = (pred_binary * targets_all).sum()
    fp = (pred_binary * (1 - targets_all)).sum()
    fn = ((1 - pred_binary) * targets_all).sum()
    tn = ((1 - pred_binary) * (1 - targets_all)).sum()
    
    smooth = 1e-6
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    sens = (tp + smooth) / (tp + fn + smooth)
    spec = (tn + smooth) / (tn + fp + smooth)
    prec = (tp + smooth) / (tp + fp + smooth)
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    # AUC metrics
    auc_roc = compute_auc_roc(pred_probs_all, targets_all)
    auc_pr = compute_auc_pr(pred_probs_all, targets_all)
    
    return {
        'dice': dice,
        'iou': iou,
        'sensitivity': sens,
        'specificity': spec,
        'precision': prec,
        'accuracy': acc,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
    }
