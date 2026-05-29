"""Visualization module for retinal analysis panels.

Produces 5 panels:
  1. Original Fundus (CLAHE preprocessed)
  2. Probability Heatmap (vessel softmax, jet colormap)
  3. Binary Vessel Mask (cyan vessels on black background)
  4. Damage Analysis (vessel mask + severity-colored damage annotations)
  5. Vessel Overlay (green vessel blend on original fundus)
"""

import cv2
import numpy as np
from PIL import Image

from app.schemas.reports import DamageRegion
from app.utils.image_io import pil_to_data_url


def _rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


# ---------------------------------------------------------------------------
# Panel 2: Probability Heatmap
# ---------------------------------------------------------------------------
def render_heatmap(prob_map: np.ndarray) -> Image.Image:
    """Render probability map as a JET-colormap heatmap."""
    heat = np.clip(prob_map * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Panel 3: Colorized Vessel Mask (cyan on black)
# ---------------------------------------------------------------------------
def render_clean_mask(mask: np.ndarray) -> Image.Image:
    """Convert binary vessel mask to cyan (G+B) on black — clinical standard look."""
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
    colored = np.zeros((*mask_uint8.shape[:2], 3), dtype=np.uint8)
    colored[..., 0] = 0               # R = 0
    colored[..., 1] = mask_uint8      # G = vessel intensity
    colored[..., 2] = mask_uint8      # B = vessel intensity → cyan
    return Image.fromarray(colored)


# ---------------------------------------------------------------------------
# Panel 4: Damage Analysis (vessel mask + severity ellipses)
# ---------------------------------------------------------------------------
# Severity colors for damage annotations
SEVERITY_COLORS = {
    "low": (250, 204, 21),       # Yellow
    "medium": (245, 158, 11),    # Orange
    "high": (239, 68, 68),       # Red
}


def render_damage_mask(mask: np.ndarray, regions: list[DamageRegion]) -> Image.Image:
    """Render vessel mask with colored damage region annotations."""
    # Start with the cyan vessel mask as base
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
    canvas = np.zeros((*mask_uint8.shape[:2], 3), dtype=np.uint8)
    vessel_pixels = mask_uint8 > 0
    canvas[vessel_pixels, 1] = mask_uint8[vessel_pixels]  # Green channel
    canvas[vessel_pixels, 2] = mask_uint8[vessel_pixels]  # Blue channel → cyan

    # Draw severity-colored ellipses around damage regions
    for region in regions:
        cx = (region.x_min + region.x_max) // 2
        cy = (region.y_min + region.y_max) // 2
        rx = max(8, (region.x_max - region.x_min) // 2)
        ry = max(8, (region.y_max - region.y_min) // 2)
        color = SEVERITY_COLORS.get(region.severity, (250, 204, 21))
        cv2.ellipse(canvas, (cx, cy), (rx, ry), 0, 0, 360, color, 2)

    return Image.fromarray(canvas)


# ---------------------------------------------------------------------------
# Panel 5: Vessel Overlay (green vessels blended on fundus)
# ---------------------------------------------------------------------------
def render_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Blend vessel mask as green overlay on original fundus image."""
    base = _rgb(image).copy().astype(np.float32)
    mask_binary = mask > 0 if mask.max() <= 1 else mask > 127
    alpha = 0.45

    overlay = base.copy()
    # Dim R and B channels, boost G channel on vessel pixels
    overlay[mask_binary, 0] = np.clip(overlay[mask_binary, 0] * 0.3, 0, 255)
    overlay[mask_binary, 1] = np.clip(overlay[mask_binary, 1] * 0.6 + 120, 0, 255)
    overlay[mask_binary, 2] = np.clip(overlay[mask_binary, 2] * 0.3, 0, 255)

    blended = cv2.addWeighted(
        base.astype(np.uint8), 1 - alpha,
        overlay.astype(np.uint8), alpha,
        0,
    )
    return Image.fromarray(blended)


# ---------------------------------------------------------------------------
# Zoom crops for damage regions
# ---------------------------------------------------------------------------
def build_zoom_crops(mask: np.ndarray, regions: list[DamageRegion], crop_size: int = 128) -> list[dict]:
    source = render_damage_mask(mask, regions)
    crops: list[dict] = []
    for region in regions[:3]:
        cx = (region.x_min + region.x_max) // 2
        cy = (region.y_min + region.y_max) // 2
        half = max(region.x_max - region.x_min, region.y_max - region.y_min, 64)
        left = max(0, cx - half)
        top = max(0, cy - half)
        right = min(512, cx + half)
        bottom = min(512, cy + half)
        crop = source.crop((left, top, right, bottom)).resize(
            (crop_size, crop_size), Image.Resampling.LANCZOS
        )
        crops.append(
            {
                "image": pil_to_data_url(crop),
                "finding": region.finding,
                "severity": region.severity,
                "quadrant": region.quadrant,
            }
        )
    return crops


# ---------------------------------------------------------------------------
# Build all 5 panels
# ---------------------------------------------------------------------------
def build_panels(
    image: Image.Image,
    prob_map: np.ndarray,
    clean_mask: np.ndarray,
    regions: list[DamageRegion],
) -> dict:
    return {
        "original": pil_to_data_url(image),
        "heatmap": pil_to_data_url(render_heatmap(prob_map)),
        "vessel_clean": pil_to_data_url(render_clean_mask(clean_mask)),
        "vessel_annotated": pil_to_data_url(render_damage_mask(clean_mask, regions)),
        "overlay": pil_to_data_url(render_overlay(image, clean_mask)),
    }
