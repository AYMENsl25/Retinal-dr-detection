"""
Retinal DR Detection — Lesion Segmentation Dataset
=====================================================
Multi-label dataset for simultaneous segmentation of 5 lesion types.

Key difference from VesselDataset (src/dataset.py):
  - Vessel segmentation = SINGLE binary mask  → model outputs (B, 1, H, W)
  - Lesion segmentation = FIVE  binary masks  → model outputs (B, 5, H, W)

Each image produces a stacked mask tensor of shape (5, H, W) where:
  Channel 0 → MA  (Microaneurysms)        — earliest sign of DR
  Channel 1 → HE  (Hemorrhages)           — blood leaking from vessels
  Channel 2 → EX  (Hard Exudates)         — lipid deposits, sign of macular edema
  Channel 3 → OD  (Optic Disc)            — anatomical landmark for spatial reference
  Channel 4 → CW  (Cotton Wool spots)     — nerve fiber infarcts, severe ischemia

Albumentations automatically applies the SAME spatial transform (flip,
rotate, distort) to ALL mask channels, ensuring geometric consistency.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# The 5 lesion types — this is the SINGLE source of truth
# ============================================================
LESION_TYPES = ['MA', 'HE', 'EX', 'OD', 'CW']
NUM_LESION_CLASSES = len(LESION_TYPES)


class LesionDataset(Dataset):
    """
    Multi-label retinal lesion segmentation dataset.

    For each fundus image it loads 5 binary masks (one per lesion type)
    and stacks them into a single (5, H, W) tensor.  If a mask file
    doesn't exist for a particular lesion type, a zero-mask (= "no
    lesion present") is used — this is the correct ground truth.

    Args:
        csv_file     : Path to CSV with 'img_id' and 'img_path' columns.
        base_dir     : Root directory of the processed dataset
                       (e.g. 'dataset_stage1_segmentation_processed').
        transform    : Albumentations Compose pipeline (from get_*_transforms).
        lesion_mask_dir : Sub-folder containing per-type mask folders
                         (default 'lesion_masks').
        lesion_types : Which lesion types to load (default LESION_TYPES).
    """

    def __init__(self, csv_file, base_dir, transform=None,
                 lesion_mask_dir='lesion_masks',
                 lesion_types=None,
                 blacklist_path=None):
        self.df = pd.read_csv(csv_file)

        # Keep only rows that have at least one lesion mask
        if 'has_lesion' in self.df.columns:
            self.df = self.df[self.df['has_lesion'] == True].reset_index(drop=True)

        # ── Blacklist filtering: remove corrupted/low-quality images ──
        blacklisted_ids = self._load_blacklist(blacklist_path)
        if blacklisted_ids:
            before_count = len(self.df)
            self.df = self.df[~self.df['img_id'].isin(blacklisted_ids)].reset_index(drop=True)
            removed = before_count - len(self.df)
            if removed > 0:
                print(f"🚫 Blacklist: removed {removed} corrupted images "
                      f"({before_count} → {len(self.df)})")

        self.base_dir = base_dir
        self.lesion_mask_dir = lesion_mask_dir
        self.transform = transform
        self.lesion_types = lesion_types or LESION_TYPES

    @staticmethod
    def _load_blacklist(path=None):
        """
        Load blacklisted image IDs from a text file.

        Searches for 'blacklisted_images.txt' in common locations:
          1. Explicit path argument
          2. Project root (auto-detected from src/lesion/ location)

        File format: one img_id per line, '#' comments ignored.

        Returns:
            set of blacklisted image IDs, or empty set
        """
        search_paths = []
        if path:
            search_paths.append(path)

        # Auto-detect project root (two levels up from this file)
        import pathlib
        module_dir = pathlib.Path(__file__).resolve().parent
        project_root = module_dir.parent.parent
        search_paths.append(str(project_root / 'blacklisted_images.txt'))

        for p in search_paths:
            if os.path.exists(p):
                with open(p, 'r') as f:
                    ids = set()
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            ids.add(line)
                if ids:
                    print(f"📋 Loaded {len(ids)} blacklisted IDs from {p}")
                return ids

        return set()

    def __len__(self):
        return len(self.df)

    # ----------------------------------------------------------
    def _load_single_mask(self, img_id, lesion_type):
        """
        Attempt to load one lesion mask from disk.

        Path convention (set during preprocessing):
            {base_dir}/lesion_masks/{lesion_type}/{img_id}
            e.g.  dataset_processed/lesion_masks/MA/IDRID_train_IDRiD_01.png

        Returns:
            np.ndarray (H, W) float32 with values 0.0 / 1.0, or None.
        """
        mask_path = os.path.join(
            self.base_dir, self.lesion_mask_dir, lesion_type, img_id
        )
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return (mask > 127).astype(np.float32)
        return None

    # ----------------------------------------------------------
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['img_id']               # e.g. "IDRID_train_IDRiD_01.png"

        # ---- 1. Load RGB image ----
        img_path = os.path.join(self.base_dir, row['img_path'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        # ---- 2. Load & stack all 5 lesion masks → (H, W, 5) ----
        #   Albumentations treats a 3-D mask array as a multi-channel
        #   mask and applies the same geometric transform to every channel.
        mask_stack = np.zeros((h, w, len(self.lesion_types)), dtype=np.float32)
        for i, lt in enumerate(self.lesion_types):
            m = self._load_single_mask(img_id, lt)
            if m is not None:
                mask_stack[:, :, i] = m

        # ---- 3. Apply augmentations ----
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask_stack)
            image = augmented['image']       # Tensor (3, H, W) after ToTensorV2
            mask_stack = augmented['mask']    # Tensor (5, H, W) after ToTensorV2

        # ---- 4. Ensure correct tensor format: (C, H, W) ----
        #   ToTensorV2 in newer albumentations does NOT auto-transpose
        #   multi-channel masks, so we handle it explicitly.
        if isinstance(mask_stack, np.ndarray):
            # No transform applied — convert manually
            mask_tensor = torch.from_numpy(
                mask_stack.transpose(2, 0, 1)  # (H, W, 5) → (5, H, W)
            ).float()
        elif isinstance(mask_stack, torch.Tensor):
            if mask_stack.dim() == 3 and mask_stack.shape[-1] == len(self.lesion_types):
                # Shape is (H, W, 5) — need to permute to (5, H, W)
                mask_tensor = mask_stack.permute(2, 0, 1).float()
            else:
                mask_tensor = mask_stack.float()
        else:
            mask_tensor = torch.as_tensor(mask_stack).float()

        return image, mask_tensor


# ============================================================
# Augmentation Pipelines (same as vessel, reused here)
# ============================================================

def get_train_transforms(img_size=512):
    """
    Training augmentation pipeline.

    Geometric transforms handle fundus image variability:
      - Different camera angles       → flips & rotations
      - Different fields of view      → scale variations
      - Patient movement              → shift & distortion

    Color transforms handle:
      - Different fundus camera brands → brightness/contrast variation
      - Different illumination         → hue/saturation shifts
    """
    return A.Compose([
        A.Resize(img_size, img_size),

        # === Geometric Augmentations ===
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={'x': (-0.0625, 0.0625), 'y': (-0.0625, 0.0625)},
            scale=(0.9, 1.1),
            rotate=(-45, 45),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.2),

        # === Color / Illumination Augmentations ===
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.2, hue=0.1, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.02, 0.05), p=0.2),

        # === Normalization (ImageNet stats for pretrained backbones) ===
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=512):
    """
    Validation/Test transform — NO augmentation, only resize + normalize.
    We must evaluate on clean, unaugmented images for fair comparison.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
