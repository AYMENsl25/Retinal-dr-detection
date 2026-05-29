import cv2
import numpy as np
from skimage.morphology import skeletonize


def _quadrant(cx: int, cy: int, size: int = 512) -> str:
    if cy < size // 2 and cx < size // 2:
        return "NW"
    if cy < size // 2:
        return "NE"
    if cx < size // 2:
        return "SW"
    return "SE"


def _severity(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def _bbox_from_point(x: int, y: int, radius: int = 28, size: int = 512) -> dict:
    return {
        "x_min": max(0, x - radius),
        "y_min": max(0, y - radius),
        "x_max": min(size, x + radius),
        "y_max": min(size, y + radius),
    }


def analyze_vessel_damage(mask: np.ndarray, size: int = 512) -> dict:
    """Find deterministic damage candidates from a clean vessel mask.

    The LLM should explain and rank these regions, not invent coordinates from
    nothing. This keeps red ellipses grounded in image processing evidence.
    """

    binary = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary > 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    neighbors = cv2.filter2D(skeleton, -1, kernel) - skeleton
    endpoints = np.argwhere((skeleton == 1) & (neighbors == 1))
    branchpoints = np.argwhere((skeleton == 1) & (neighbors >= 3))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
    component_scores: list[tuple[float, int, int, int, int, int]] = []
    tortuosities: list[float] = []

    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 10:
            continue
        chord = max(1.0, float(np.hypot(w, h)))
        tortuosity = float(area / chord)
        tortuosities.append(tortuosity)
        compactness = min(1.0, tortuosity / 4.0)
        small_fragment = 1.0 if area < 45 else 0.0
        score = 0.55 * compactness + 0.45 * small_fragment
        component_scores.append((score, x, y, w, h, area))

    endpoint_regions = []
    for y, x in endpoints[:24]:
        nearby_branchpoints = np.linalg.norm(branchpoints - np.array([y, x]), axis=1) if len(branchpoints) else []
        isolated = len(nearby_branchpoints) == 0 or float(np.min(nearby_branchpoints)) > 18
        if isolated:
            endpoint_regions.append((0.65, int(x), int(y)))

    regions: list[dict] = []
    for score, x, y, w, h, area in sorted(component_scores, reverse=True)[:5]:
        cx = x + w // 2
        cy = y + h // 2
        pad = 18
        regions.append(
            {
                "x_min": max(0, x - pad),
                "y_min": max(0, y - pad),
                "x_max": min(size, x + w + pad),
                "y_max": min(size, y + h + pad),
                "quadrant": _quadrant(cx, cy, size),
                "severity": _severity(score),
                "finding": "Tortuous or fragmented segment",
                "score": round(float(score), 3),
            }
        )

    for score, x, y in endpoint_regions[:3]:
        regions.append(
            {
                **_bbox_from_point(x, y, 26, size),
                "quadrant": _quadrant(x, y, size),
                "severity": _severity(score),
                "finding": "Possible vessel discontinuity",
                "score": round(float(score), 3),
            }
        )

    top_regions = sorted(regions, key=lambda region: region["score"], reverse=True)[:8]
    mean_tortuosity = float(np.mean(tortuosities)) if tortuosities else 1.0
    broken_segments = min(99, len(endpoint_regions))

    return {
        "skeleton_endpoints": int(len(endpoints)),
        "branchpoints": int(len(branchpoints)),
        "mean_tortuosity": round(mean_tortuosity, 3),
        "broken_segments_estimate": int(broken_segments),
        "candidate_regions": top_regions,
    }
