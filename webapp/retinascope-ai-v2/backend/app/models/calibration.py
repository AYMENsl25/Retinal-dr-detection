import math

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    safe_temperature = max(temperature, 1e-6)
    scaled = logits.astype(np.float64) / safe_temperature
    shifted = scaled - np.max(scaled)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def predictive_entropy(probs: np.ndarray) -> float:
    clipped = np.clip(probs.astype(np.float64), 1e-9, 1.0)
    return round(float(-np.sum(clipped * np.log(clipped)) / math.log(len(probs))), 3)
