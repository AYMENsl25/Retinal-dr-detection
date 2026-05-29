import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def crop_roi(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crop the region of interest from a fundus image, removing black borders.

    This matches the exact preprocessing used during model training.
    """
    if img.ndim == 2:
        mask = img > tol
    elif img.ndim == 3:
        mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > tol
    else:
        return img
    m, n = mask.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    if not mask0.any() or not mask1.any():
        return img
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


DEFAULT_PREPROCESS_CONFIG = {
    "input_size": [512, 512],
    "color_space": "RGB",
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
    "clahe": {"enabled": False},
    "vessel": {"threshold": 0.5, "input_channels": 3},
    "grader": {
        "input_channels": 9,
        "mask_channels": ["vessel", "MA", "HE", "EX", "OD", "CW"],
        "image_channels": {
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "mask_channels_normalization": {
            "normalize_mean": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "normalize_std": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        },
    },
}


def load_preprocess_config(path: str | Path | None) -> dict:
    if not path:
        return DEFAULT_PREPROCESS_CONFIG
    config_path = Path(path)
    if not config_path.exists():
        return DEFAULT_PREPROCESS_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return {**DEFAULT_PREPROCESS_CONFIG, **loaded}


def clahe_rgb(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.merge((enhanced_l, a_channel, b_channel))
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB))


def prepare_retina_image(image: Image.Image, config: dict) -> Image.Image:
    image_size = config.get("input_size", [512, 512])
    prepared = image.convert("RGB").resize(tuple(image_size), Image.Resampling.LANCZOS)
    if bool(config.get("clahe", {}).get("enabled", False)):
        return clahe_rgb(prepared)
    return prepared


def normalize_rgb_array(image: Image.Image, config: dict) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    mean = np.asarray(config.get("normalize_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.asarray(config.get("normalize_std", [0.229, 0.224, 0.225]), dtype=np.float32)
    return (rgb - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)


def build_grader_input_array(
    image: Image.Image,
    vessel_mask: np.ndarray,
    config: dict,
    lesion_masks: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Create the 9-channel grader input: RGB + vessel + lesion masks.

    If lesion masks are unavailable, they are filled with zeros. That keeps the
    tensor shape valid but may reduce grader accuracy if the checkpoint was
    trained with real MA/HE/EX/OD/CW masks.
    """

    grader = config.get("grader", {})
    image_config = grader.get("image_channels", {})
    rgb_config = {
        **config,
        "normalize_mean": image_config.get("normalize_mean", config.get("normalize_mean")),
        "normalize_std": image_config.get("normalize_std", config.get("normalize_std")),
    }
    rgb = normalize_rgb_array(image, rgb_config)

    lesion_masks = lesion_masks or {}
    mask_names = grader.get("mask_channels", ["vessel", "MA", "HE", "EX", "OD", "CW"])
    mask_norm = grader.get("mask_channels_normalization", {})
    mask_means = mask_norm.get("normalize_mean", [0.5] * len(mask_names))
    mask_stds = mask_norm.get("normalize_std", [0.5] * len(mask_names))

    masks: list[np.ndarray] = []
    for index, name in enumerate(mask_names):
        if name == "vessel":
            mask = vessel_mask
        else:
            mask = lesion_masks.get(name, np.zeros_like(vessel_mask))
        mask_float = np.asarray(mask, dtype=np.float32)
        if mask_float.max(initial=0) > 1.0:
            mask_float = mask_float / 255.0
        mean = float(mask_means[index])
        std = max(float(mask_stds[index]), 1e-6)
        masks.append((mask_float - mean) / std)

    stacked = np.dstack([rgb, *masks]).astype(np.float32)
    expected_channels = int(grader.get("input_channels", 9))
    if stacked.shape[-1] != expected_channels:
        raise ValueError(
            f"Grader input has {stacked.shape[-1]} channels, expected {expected_channels}"
        )
    return stacked


def circular_fov_mask(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    green = rgb[:, :, 1]
    _, mask = cv2.threshold(green, 12, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask.astype(np.uint8)
