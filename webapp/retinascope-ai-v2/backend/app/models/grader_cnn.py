"""Grader model loader utility.

This module provides a flexible loader that attempts to load a grader
checkpoint saved as:
 - a TorchScript/traced module
 - a full nn.Module (saved via torch.save(model))
 - a dict containing a `'model'` key with an nn.Module

State-dict checkpoints are loaded as the 9-channel ConvNeXt-Base grader used
by the Stage B training pipeline unless the checkpoint cfg says otherwise.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def load_grader_model(path: Path | str, device: torch.device) -> torch.nn.Module:
    """Try to load a grader model from `path`.

    Returns an nn.Module ready for inference on `device`.
    """
    p = str(path)

    # 1) Try TorchScript / traced model
    try:
        model = torch.jit.load(p, map_location=device)
        model.to(device)
        model.eval()
        logger.info("Loaded grader model via TorchScript from %s", p)
        return model
    except Exception:
        pass

    # 2) Try torch.load
    try:
        obj = torch.load(p, map_location=device, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load grader checkpoint {p}: {exc}")

    # If a module instance was saved directly
    if isinstance(obj, torch.nn.Module):
        obj.to(device)
        obj.eval()
        logger.info("Loaded grader nn.Module from %s", p)
        return obj

    # If a state dict or wrapper dict was saved
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], torch.nn.Module):
            obj["model"].to(device)
            obj["model"].eval()
            logger.info("Loaded grader from dict 'model' key in %s", p)
            return obj["model"]
        if "model_state_dict" in obj or "state_dict" in obj:
            import timm
            cfg = obj.get("cfg", {})
            model_name = cfg.get("model_name", "convnext_base")
            num_channels = cfg.get("num_channels", 9)
            num_classes = cfg.get("num_classes", 5)
            logger.info(
                "Instantiating model '%s' with %d channels and %d classes...",
                model_name, num_channels, num_classes
            )
            model = timm.create_model(
                model_name,
                pretrained=False,
                in_chans=num_channels,
                num_classes=num_classes,
            )
            state_dict = obj.get("model_state_dict", obj.get("state_dict"))
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            logger.info("Loaded grader model from state_dict in %s", p)
            return model

    raise RuntimeError(f"Unsupported grader checkpoint format: {p}")
