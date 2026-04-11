"""
Retinal DR Detection - Generic Training Engine
================================================
Provides a reusable training loop with:
- Mixed precision training (faster on modern GPUs)
- Learning rate scheduling (OneCycleLR / ReduceLROnPlateau)
- Early stopping (saves GPU hours when model plateaus)
- Model checkpointing (saves best model based on val Dice)
- Metric tracking per epoch for later plotting
"""

import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm.notebook import tqdm

from src.metrics import evaluate_batch


# ============================================================
# Loss Functions
# ============================================================

class DiceLoss(nn.Module):
    """
    Dice Loss = 1 - Dice Coefficient
    
    Directly optimizes the Dice score, which is our primary evaluation
    metric. Works much better than BCE alone for imbalanced segmentation.
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred_logits, targets):
        pred = torch.sigmoid(pred_logits)
        pred_flat = pred.contiguous().view(-1)
        target_flat = targets.contiguous().view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss + Binary Cross Entropy Loss.
    
    Why combine?
    - BCE provides stable gradients at the pixel level (good for learning)
    - Dice directly optimizes our evaluation metric (good for performance)
    - Together they converge faster and reach better optima.
    
    This is the most commonly used loss for medical image segmentation.
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, pred_logits, targets):
        dice = self.dice_loss(pred_logits, targets)
        bce = self.bce_loss(pred_logits, targets)
        return self.dice_weight * dice + self.bce_weight * bce


# ============================================================
# Training & Validation Steps
# ============================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """
    Train for one epoch.
    
    Args:
        model: Segmentation model
        dataloader: Training DataLoader
        optimizer: Adam/AdamW optimizer
        criterion: Loss function (DiceBCELoss)
        device: cuda/cpu
        scaler: GradScaler for mixed precision (optional)
    
    Returns:
        dict with average loss and metrics for this epoch
    """
    model.train()
    epoch_loss = 0.0
    epoch_metrics = {'dice': 0, 'iou': 0, 'sensitivity': 0, 'specificity': 0}
    num_batches = 0
    
    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        # Track metrics
        batch_metrics = evaluate_batch(outputs, masks)
        epoch_loss += loss.item()
        for k in epoch_metrics:
            epoch_metrics[k] += batch_metrics[k]
        num_batches += 1
    
    # Average over batches
    epoch_loss /= num_batches
    for k in epoch_metrics:
        epoch_metrics[k] /= num_batches
    epoch_metrics['loss'] = epoch_loss
    
    return epoch_metrics


def validate(model, dataloader, criterion, device):
    """
    Validate the model (no gradient computation).
    
    Returns:
        dict with average loss and metrics for validation set
    """
    model.eval()
    epoch_loss = 0.0
    epoch_metrics = {'dice': 0, 'iou': 0, 'sensitivity': 0, 'specificity': 0}
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            batch_metrics = evaluate_batch(outputs, masks)
            epoch_loss += loss.item()
            for k in epoch_metrics:
                epoch_metrics[k] += batch_metrics[k]
            num_batches += 1
    
    epoch_loss /= num_batches
    for k in epoch_metrics:
        epoch_metrics[k] /= num_batches
    epoch_metrics['loss'] = epoch_loss
    
    return epoch_metrics


# ============================================================
# Full Training Loop
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs=50,
    scheduler=None,
    early_stopping_patience=10,
    save_dir='checkpoints',
    model_name='model',
    use_amp=True,
):
    """
    Complete training loop with early stopping and checkpointing.
    
    Args:
        model: Segmentation model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: Optimizer (e.g., AdamW)
        criterion: Loss function
        device: 'cuda' or 'cpu'
        num_epochs: Maximum number of epochs
        scheduler: LR scheduler (optional)
        early_stopping_patience: Stop if val Dice doesn't improve for N epochs
        save_dir: Directory to save model checkpoints
        model_name: Name prefix for saved checkpoint files
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        history: dict with lists of train/val metrics per epoch
        best_model_path: path to the best saved model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    scaler = GradScaler() if (use_amp and device.type == 'cuda') else None
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'train_iou': [], 'val_iou': [],
        'train_sensitivity': [], 'val_sensitivity': [],
        'train_specificity': [], 'val_specificity': [],
    }
    
    best_val_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_path = os.path.join(save_dir, f'{model_name}_best.pth')
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} for up to {num_epochs} epochs")
    print(f"Device: {device} | AMP: {use_amp} | Patience: {early_stopping_patience}")
    print(f"{'='*60}\n")
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['dice'])
            else:
                scheduler.step()
        
        # Record history
        for key in ['loss', 'dice', 'iou', 'sensitivity', 'specificity']:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )
        
        # Check for improvement
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_val_dice,
                'history': history,
            }, best_model_path)
            print(f"  ★ New best model saved! Dice: {best_val_dice:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹ Early stopping at epoch {epoch}. "
                      f"Best Dice: {best_val_dice:.4f} at epoch {best_epoch}")
                break
    
    print(f"\n{'='*60}")
    print(f"Training complete. Best Val Dice: {best_val_dice:.4f} (epoch {best_epoch})")
    print(f"Best model saved to: {best_model_path}")
    print(f"{'='*60}")
    
    return history, best_model_path
