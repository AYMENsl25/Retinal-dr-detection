"""
YOLOv11-Seg Dataset Conversion Utility
========================================
This script converts the 5-channel binary semantic masks into 
YOLO standard polygon text files (instance segmentation).

Handles tiny lesions (Microaneurysms) by mathematically guaranteeing
valid polygon shapes for YOLO to learn from.
"""

import os
import cv2
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm

# Constants
BASE_DIR = 'data/DR_dataset_processed_segmentation'
OUT_DIR = 'data/yolo_lesion_dataset'
LESION_TYPES = ['MA', 'HE', 'EX', 'OD', 'CW']
IMG_SIZE = 512.0  # YOLO normalizes coordinates to 0.0-1.0

def create_circle_polygon(cx, cy, radius, num_points=8):
    """Creates a small mathematical circle polygon around a tiny dot."""
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        # Keep inside image bounds
        x = np.clip(x, 0, IMG_SIZE - 1)
        y = np.clip(y, 0, IMG_SIZE - 1)
        points.append(f"{x / IMG_SIZE:.6f} {y / IMG_SIZE:.6f}")
    return " ".join(points)

def process_image(img_id, source_img_path, split_name):
    """Copies image to target dir and extracts YOLO polygons from 5 masks."""
    
    # Define destination folders based on CSV split
    out_img_dir = os.path.join(OUT_DIR, split_name, 'images')
    out_label_dir = os.path.join(OUT_DIR, split_name, 'labels')
    
    # 1. Copy image over to YOLO structure
    src_img = os.path.join(BASE_DIR, source_img_path)
    dst_img = os.path.join(out_img_dir, img_id)
    if os.path.exists(src_img):
        shutil.copy2(src_img, dst_img)
    
    # The label file starts empty. If no lesions are found, it stays empty (Background Image).
    label_path = os.path.join(out_label_dir, img_id.replace('.png', '.txt').replace('.jpg', '.txt'))
    
    yolo_lines = []
    
    # 2. Process all 5 lesion types
    for class_id, lt in enumerate(LESION_TYPES):
        mask_path = os.path.join(BASE_DIR, 'lesion_masks', lt, img_id)
        if not os.path.exists(mask_path):
            continue
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        # Extract boundaries of every white blob
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            # If the blob is extremely tiny (e.g. 1 point or line), draw a math circle
            if cnt.shape[0] < 3 or cv2.contourArea(cnt) < 4.0:
                # Find the center of the tiny dot
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                else:
                    cx, cy = cnt[0][0]
                
                # Draw a tiny polygon so YOLO can see it (radius=3 pixels padding)
                poly_str = create_circle_polygon(cx, cy, radius=3)
                yolo_lines.append(f"{class_id} {poly_str}")
            
            else:
                # Standard Polygon calculation
                points = []
                for pt in cnt:
                    x, y = pt[0]
                    # Normalize against 512.0
                    points.append(f"{x / IMG_SIZE:.6f} {y / IMG_SIZE:.6f}")
                
                poly_str = " ".join(points)
                yolo_lines.append(f"{class_id} {poly_str}")
                
    # 3. Write label file
    with open(label_path, 'w') as f:
        f.write("\n".join(yolo_lines))


def main():
    # Setup directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUT_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, split, 'labels'), exist_ok=True)
        
    print("Starting YOLO dataset conversion...")
    
    splits = {
        'train': pd.read_csv(os.path.join(BASE_DIR, 'train_split.csv')),
        'val':   pd.read_csv(os.path.join(BASE_DIR, 'val_split.csv')),
        'test':  pd.read_csv(os.path.join(BASE_DIR, 'test_split.csv'))
    }
    
    # Process images split by split
    for split_name, df in splits.items():
        print(f"Processing {split_name} (Total: {len(df)} images)...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            process_image(row['img_id'], row['img_path'], split_name)
            
    # Write data.yaml config for YOLO
    yaml_content = f"""path: ../yolo_lesion_dataset  # Keep relative or absolute path based on colab mount
train: train/images
val: val/images
test: test/images

nc: 5
names:
  0: MA    # Microaneurysms
  1: HE    # Hemorrhages
  2: EX    # Hard Exudates
  3: OD    # Optic Disc
  4: CW    # Cotton Wool Spots
"""
    with open(os.path.join(OUT_DIR, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
        
    print("✅ YOLO Dataset created successfully at:", OUT_DIR)

if __name__ == "__main__":
    main()
