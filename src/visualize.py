"""
Retinal DR Detection - Visualization Utilities
================================================
Provides functions for:
- Training history curves (loss, Dice, IoU over epochs)
- Prediction overlays (original image + ground truth + model prediction)
- Side-by-side model comparison charts
- Confusion matrix display
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, precision_recall_curve


# ============================================================
# Training History Plots
# ============================================================

def plot_training_history(history, model_name='Model'):
    """
    Plot loss and all key metrics over training epochs.
    
    Creates a 2x2 grid:
    - Top-left: Loss (train vs val)
    - Top-right: Dice (train vs val)
    - Bottom-left: IoU (train vs val)
    - Bottom-right: Sensitivity & Specificity (val only)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} — Training History', fontsize=16, fontweight='bold')
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0, 0].set_title('Loss (DiceBCE)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dice
    axes[0, 1].plot(epochs, history['train_dice'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['val_dice'], 'r-', label='Val')
    axes[0, 1].set_title('Dice Coefficient')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # IoU
    axes[1, 0].plot(epochs, history['train_iou'], 'b-', label='Train')
    axes[1, 0].plot(epochs, history['val_iou'], 'r-', label='Val')
    axes[1, 0].set_title('IoU (Jaccard)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sensitivity & Specificity
    axes[1, 1].plot(epochs, history['val_sensitivity'], 'g-', label='Val Sensitivity')
    axes[1, 1].plot(epochs, history['val_specificity'], 'm-', label='Val Specificity')
    axes[1, 1].set_title('Sensitivity & Specificity (Val)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    return fig


# ============================================================
# Prediction Visualization
# ============================================================

def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor.cpu() * std + mean
    return torch.clamp(tensor, 0, 1).permute(1, 2, 0).numpy()


def plot_predictions(model, dataloader, device, num_samples=4, threshold=0.5, model_name='Model'):
    """
    Visualize model predictions on validation/test data.
    
    For each sample shows 4 columns:
    1. Original fundus image
    2. Ground truth vessel mask
    3. Model's predicted mask
    4. Overlay (original + prediction in red + ground truth in green)
    """
    model.eval()
    images_shown = 0
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    fig.suptitle(f'{model_name} — Predictions', fontsize=16, fontweight='bold')
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    col_titles = ['Original Image', 'Ground Truth', 'Prediction', 'Overlay']
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12, fontweight='bold')
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds_binary = (preds > threshold).float()
            
            for i in range(images.shape[0]):
                if images_shown >= num_samples:
                    break
                    
                img_np = denormalize(images[i])
                gt_np = masks[i].squeeze().cpu().numpy()
                pred_np = preds_binary[i].squeeze().cpu().numpy()
                
                # Original image
                axes[images_shown, 0].imshow(img_np)
                axes[images_shown, 0].axis('off')
                
                # Ground truth
                axes[images_shown, 1].imshow(gt_np, cmap='gray')
                axes[images_shown, 1].axis('off')
                
                # Prediction
                axes[images_shown, 2].imshow(pred_np, cmap='gray')
                axes[images_shown, 2].axis('off')
                
                # Overlay: Green = GT, Red = Prediction
                overlay = img_np.copy()
                overlay[gt_np > 0.5] = [0, 1, 0]      # Green for ground truth
                overlay[pred_np > 0.5] = [1, 0, 0]     # Red for prediction
                # Yellow where both overlap (correct predictions)
                both = (gt_np > 0.5) & (pred_np > 0.5)
                overlay[both] = [1, 1, 0]
                axes[images_shown, 3].imshow(overlay)
                axes[images_shown, 3].axis('off')
                
                images_shown += 1
                
            if images_shown >= num_samples:
                break
    
    plt.tight_layout()
    plt.show()
    return fig


# ============================================================
# ROC and PR Curve Plots
# ============================================================

def plot_roc_pr_curves(pred_probs, targets, model_name='Model'):
    """
    Plot ROC curve and Precision-Recall curve side by side.
    
    Args:
        pred_probs: Flattened numpy array of sigmoid probabilities
        targets: Flattened numpy array of binary ground truth
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} — ROC & PR Curves', fontsize=14, fontweight='bold')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(targets, pred_probs)
    ax1.plot(fpr, tpr, 'b-', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.grid(True, alpha=0.3)
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(targets, pred_probs)
    ax2.plot(recall, precision, 'r-', linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    return fig


# ============================================================
# Model Comparison Table
# ============================================================

def print_comparison_table(results_dict):
    """
    Print a formatted comparison table of all models.
    
    Args:
        results_dict: dict of {model_name: {metric_name: value}}
    
    Example:
        results = {
            'U-Net': {'dice': 0.82, 'iou': 0.70, ...},
            'TransUNet': {'dice': 0.85, 'iou': 0.74, ...},
        }
    """
    metrics = ['dice', 'iou', 'sensitivity', 'specificity', 'precision', 'accuracy', 'auc_roc', 'auc_pr']
    header = f"{'Model':<20}" + "".join([f"{m:>14}" for m in metrics])
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    
    for model_name, scores in results_dict.items():
        row = f"{model_name:<20}"
        for m in metrics:
            val = scores.get(m, 0)
            row += f"{val:>14.4f}"
        print(row)
    
    print("=" * len(header))
    
    # Highlight best model per metric
    print("\n★ Best per metric:")
    for m in metrics:
        best_model = max(results_dict, key=lambda x: results_dict[x].get(m, 0))
        best_val = results_dict[best_model].get(m, 0)
        print(f"  {m:>14}: {best_model} ({best_val:.4f})")


# ============================================================
# Save All Results to Disk
# ============================================================

def save_all_results(history, model, dataloader, device, all_preds, all_targets,
                     test_results, save_dir='results', num_samples=4, threshold=0.5,
                     model_name='model'):
    """
    Saves EVERYTHING to the results/ folder in one call:
    - JSON metrics file (Dice, IoU, Sensitivity, etc.)
    - Training history plot (Loss, Dice, IoU curves)
    - Prediction overlays (Original / GT / Prediction / Overlay)
    - ROC & Precision-Recall curves

    Output files (example for model_name='unet_resnet34'):
        results/unet_resnet34_results.json
        results/unet_resnet34_training_history.png
        results/unet_resnet34_predictions.png
        results/unet_resnet34_roc_pr.png

    Args:
        history: dict returned by train_model()
        model: trained model (already loaded with best weights)
        dataloader: test DataLoader
        device: cuda / cpu
        all_preds: flattened numpy array of sigmoid probabilities
        all_targets: flattened numpy array of binary ground truth
        test_results: dict of metric scores from evaluate_full()
        save_dir: folder to save into (default 'results')
        num_samples: how many test images to visualize
        threshold: binarization threshold
        model_name: prefix for saved filenames
    """
    import os
    import json
    os.makedirs(save_dir, exist_ok=True)

    # --- 0. JSON Metrics ---
    path_json = os.path.join(save_dir, f'{model_name}_results.json')
    with open(path_json, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    print(f'  ✓ Saved: {path_json}')

    # --- 1. Training History ---
    fig_hist = plot_training_history(history, model_name=model_name)
    path_hist = os.path.join(save_dir, f'{model_name}_training_history.png')
    fig_hist.savefig(path_hist, dpi=150, bbox_inches='tight')
    plt.close(fig_hist)
    print(f'  ✓ Saved: {path_hist}')

    # --- 2. Prediction Overlays ---
    fig_pred = plot_predictions(model, dataloader, device,
                                num_samples=num_samples,
                                threshold=threshold,
                                model_name=model_name)
    path_pred = os.path.join(save_dir, f'{model_name}_predictions.png')
    fig_pred.savefig(path_pred, dpi=150, bbox_inches='tight')
    plt.close(fig_pred)
    print(f'  ✓ Saved: {path_pred}')

    # --- 3. ROC & PR Curves ---
    fig_roc = plot_roc_pr_curves(all_preds, all_targets, model_name=model_name)
    path_roc = os.path.join(save_dir, f'{model_name}_roc_pr.png')
    fig_roc.savefig(path_roc, dpi=150, bbox_inches='tight')
    plt.close(fig_roc)
    print(f'  ✓ Saved: {path_roc}')

    print(f'\n  All results saved to {save_dir}/')

