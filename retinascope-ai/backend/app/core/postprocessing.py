import numpy as np
from PIL import Image


def compute_mock_vessel_probability(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    green = arr[:, :, 1]
    inv = 1.0 - green

    padded = np.pad(inv, 1, mode="edge")
    sharp = (
        9 * padded[1:-1, 1:-1]
        - padded[:-2, :-2]
        - padded[:-2, 1:-1]
        - padded[:-2, 2:]
        - padded[1:-1, :-2]
        - padded[1:-1, 2:]
        - padded[2:, :-2]
        - padded[2:, 1:-1]
        - padded[2:, 2:]
    )
    prob = np.clip(sharp * 0.18 + inv * 0.4, 0, 1)
    prob *= field_of_view_weight(prob.shape[1], prob.shape[0])
    return prob.astype(np.float32)


def field_of_view_weight(width: int, height: int) -> np.ndarray:
    y, x = np.ogrid[:height, :width]
    cx = width / 2
    cy = height / 2
    radius = min(width, height) * 0.48
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / radius
    weight = 1 - np.power(distance, 4)
    return np.clip(weight, 0, 1).astype(np.float32)


def binary_mask(probability: np.ndarray, threshold: float = 0.32) -> np.ndarray:
    return (probability > threshold).astype(np.uint8)


def grade_from_biomarkers(vessel_density: float, tortuosity: float) -> int:
    score = vessel_density * 160 + (tortuosity - 1.0) * 30
    if score < 23:
        return 0
    if score < 31:
        return 1
    if score < 39:
        return 2
    if score < 48:
        return 3
    return 4


def grade_probabilities(grade: int) -> list[float]:
    raw = np.array([np.exp(-abs(idx - grade) * 1.6) for idx in range(5)], dtype=np.float32)
    raw[grade] += 0.18
    probs = raw / raw.sum()
    return [float(round(v, 6)) for v in probs]

