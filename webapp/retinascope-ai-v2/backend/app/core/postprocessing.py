import cv2
import numpy as np


def clean_binary_mask(mask: np.ndarray, min_component_area: int = 25) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            cleaned[labels == label] = 1
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    return (cleaned * 255).astype(np.uint8)


def clamp_box(box: dict, size: int = 512) -> dict:
    x_min = max(0, min(size, int(box["x_min"])))
    y_min = max(0, min(size, int(box["y_min"])))
    x_max = max(0, min(size, int(box["x_max"])))
    y_max = max(0, min(size, int(box["y_max"])))
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min
    return {**box, "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
