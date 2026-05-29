import cv2
import numpy as np


def _quadrant_density(binary: np.ndarray) -> dict[str, float]:
    h, w = binary.shape
    halves = {
        "NW": binary[: h // 2, : w // 2],
        "NE": binary[: h // 2, w // 2 :],
        "SW": binary[h // 2 :, : w // 2],
        "SE": binary[h // 2 :, w // 2 :],
    }
    return {name: round(float(region.mean()), 3) for name, region in halves.items()}


def _box_count_fractal_dimension(binary: np.ndarray) -> float:
    data = binary > 0
    min_size = min(data.shape)
    sizes = 2 ** np.arange(int(np.log2(min_size)), 3, -1)
    counts: list[int] = []
    valid_sizes: list[int] = []

    for size in sizes:
        h = data.shape[0] // size * size
        w = data.shape[1] // size * size
        if h == 0 or w == 0:
            continue
        cropped = data[:h, :w]
        blocks = cropped.reshape(h // size, size, w // size, size)
        count = np.count_nonzero(blocks.any(axis=(1, 3)))
        if count > 0:
            counts.append(int(count))
            valid_sizes.append(int(size))

    if len(counts) < 2:
        return 1.0

    coeffs = np.polyfit(np.log(1 / np.array(valid_sizes)), np.log(np.array(counts)), 1)
    return round(float(coeffs[0]), 3)


def compute_vessel_biomarkers(mask: np.ndarray, tortuosity: float, broken_segments: int) -> dict:
    binary = (mask > 0).astype(np.uint8)
    count, _ = cv2.connectedComponents(binary, connectivity=8)
    vessel_density = round(float(binary.mean()), 3)
    return {
        "vessel_density": vessel_density,
        "tortuosity": round(float(tortuosity), 3),
        "fractal_dim": _box_count_fractal_dimension(binary),
        "avr": 0.75,
        "num_vessel_components": max(0, int(count - 1)),
        "num_broken_segments_estimate": int(broken_segments),
        "quadrant_density": _quadrant_density(binary),
    }
