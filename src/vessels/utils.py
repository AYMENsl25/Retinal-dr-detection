"""
Retinal DR Detection — Utility Functions
==========================================
Shared helpers used by both vessel and lesion segmentation notebooks.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Lock ALL sources of randomness so that re-running a notebook
    with IDENTICAL data + hyperparameters produces IDENTICAL results.

    Why every line matters:
      - random.seed          → Python's built-in random (used by some augmentations)
      - PYTHONHASHSEED       → hash() ordering across Python processes
      - np.random.seed       → NumPy random (used by Albumentations internally)
      - torch.manual_seed    → PyTorch CPU random
      - torch.cuda.manual_seed_all → PyTorch GPU random (all GPUs)
      - cudnn.deterministic  → Forces cuDNN to use deterministic algorithms
      - cudnn.benchmark      → Disables cuDNN auto-tuner (trades ~5-10% speed
                                for perfect reproducibility)
      - use_deterministic_algorithms → Catches ANY non-deterministic op at runtime
      - CUBLAS_WORKSPACE_CONFIG      → Required for CUDA matmul determinism

    Trade-off: Training will be ~5-10% slower because cuDNN cannot
    pick the fastest (non-deterministic) convolution algorithm.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Required for CUDA determinism
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enforce deterministic algorithms globally — raises error if any
    # non-deterministic operation is attempted (helps catch hidden issues)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Some PyTorch versions don't support this fully
        pass

    print(f'🔒 All random seeds locked to {seed} (fully reproducible)')


def worker_init_fn(worker_id):
    """
    Initialise random seeds for DataLoader workers.

    Without this, each worker uses the same NumPy seed, causing
    identical augmentations across workers in the same epoch.
    Must be passed to DataLoader: DataLoader(..., worker_init_fn=worker_init_fn)
    """
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)
