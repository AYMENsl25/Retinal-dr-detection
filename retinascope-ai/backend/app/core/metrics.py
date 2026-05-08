import numpy as np


def compute_biomarkers(mask: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    vessel_density = float(mask.mean())
    gradient_y, gradient_x = np.gradient(probability)
    edge_energy = float(np.mean(np.sqrt(gradient_x**2 + gradient_y**2)))

    tortuosity = 1.05 + min(0.65, edge_energy * 9.0)
    fractal_dim = 1.55 + min(0.2, vessel_density * 0.65)
    avr = 0.72 - min(0.14, max(0.0, vessel_density - 0.14) * 0.7)

    return {
        "vessel_density": round(vessel_density, 3),
        "tortuosity": round(tortuosity, 3),
        "fractal_dim": round(fractal_dim, 3),
        "avr": round(avr, 3),
    }

