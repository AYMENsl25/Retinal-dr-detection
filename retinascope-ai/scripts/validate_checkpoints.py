from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "backend" / "checkpoints"

EXPECTED = [
    "vessel_unet.pth",
    "lesion_unet.pth",
    "grader_cnn.pth",
    "preprocess.json",
]


def main() -> int:
    print(f"Checkpoint directory: {CHECKPOINTS}")
    missing: list[str] = []
    for name in EXPECTED:
        path = CHECKPOINTS / name
        status = "FOUND" if path.exists() else "MISSING"
        print(f"{status:8} {name}")
        if not path.exists():
            missing.append(name)

    preprocess = CHECKPOINTS / "preprocess.json"
    if preprocess.exists():
        data = json.loads(preprocess.read_text(encoding="utf-8"))
        required = ["input_size", "normalize_mean", "normalize_std", "color_space"]
        for key in required:
            if key not in data:
                print(f"MISSING  preprocess.json key: {key}")
                missing.append(f"preprocess.json:{key}")

    if missing:
        print("\nAdd the missing files before enabling real inference.")
        return 1

    print("\nAll checkpoint placeholders are present. Next: verify architecture classes load them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

