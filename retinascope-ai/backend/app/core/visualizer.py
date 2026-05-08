from base64 import b64encode
from io import BytesIO

import numpy as np
from PIL import Image


def image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def probability_to_heatmap(probability: np.ndarray) -> Image.Image:
    v = np.clip(probability, 0, 1)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask * 255).astype(np.uint8), mode="L").convert("RGB")


def red_overlay(original: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.asarray(original.convert("RGB"), dtype=np.float32)
    active = mask.astype(bool)
    arr[active, 0] = arr[active, 0] * 0.25 + 255 * 0.75
    arr[active, 1] = arr[active, 1] * 0.25
    arr[active, 2] = arr[active, 2] * 0.25
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def build_panel_data_urls(original: Image.Image, probability: np.ndarray, mask: np.ndarray) -> dict[str, str]:
    return {
        "original": image_to_data_url(original),
        "mask": image_to_data_url(mask_to_image(mask)),
        "heatmap": image_to_data_url(probability_to_heatmap(probability)),
        "overlay": image_to_data_url(red_overlay(original, mask)),
    }

