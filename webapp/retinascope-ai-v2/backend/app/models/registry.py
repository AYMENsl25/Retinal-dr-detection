"""Model registry: loads and runs the model inference pipeline.

Uses three models ensembled for vessel segmentation:
  - CSNet (0.4 weight)
  - AttentionUNet via segmentation_models_pytorch (0.4 weight)
  - SwinUNet (0.2 weight)

The grader CNN expects the same 9-channel input used when it was trained:
preprocessed RGB, vessel probability, and five lesion probability maps.
"""

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from app.config import get_settings
from app.core.postprocessing import clean_binary_mask
from app.core.preprocessing import build_grader_input_array, crop_roi, load_preprocess_config
from app.models.calibration import predictive_entropy, softmax

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    prepared_image: Image.Image
    prob_map: np.ndarray
    clean_mask: np.ndarray
    grade_probs: tuple[float, float, float, float, float]
    entropy: float
    mc_dropout_std: float


class ModelRegistry:
    """Loads the 3-model vessel segmentation ensemble and runs inference."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.vessel_loaded = False
        self.grader_loaded = False
        self.using_fallback = True
        self.preprocess_config = load_preprocess_config(self.settings.vessel_preprocess_path)

        # Device
        if self.settings.model_device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.settings.model_device)

        # Vessel models
        self.vessel_csnet = None
        self.vessel_attn = None
        self.vessel_swin = None

        # Lesion & grader models
        self.lesion_model = None
        self.lesion_attention_model = None
        self.lesion_unetpp_model = None
        self.lesion_yolo_model = None
        self.grader_model = None

        self._load_vessel_models()
        # Load grader and lesion models (optional)
        self._load_grader_and_lesion_models()

    def _load_vessel_models(self) -> None:
        """Load all three vessel segmentation models."""
        checkpoint_dir = Path(self.settings.vessel_model_path).parent

        csnet_path = checkpoint_dir / "csnet_best.pt"
        attn_path = checkpoint_dir / "attentionUNet_best_model.pt"
        swin_path = checkpoint_dir / "swin_unet_best.pt"

        try:
            # 1. CSNet
            if csnet_path.exists():
                from app.models.vessel_unet import CSNet
                self.vessel_csnet = CSNet(out_channels=1).to(self.device)
                state = torch.load(str(csnet_path), map_location=self.device, weights_only=False)
                self.vessel_csnet.load_state_dict(state["model_state_dict"])
                self.vessel_csnet.eval()
                logger.info("Loaded CSNet from %s", csnet_path)
            else:
                logger.warning("CSNet checkpoint not found: %s", csnet_path)

            # 2. AttentionUNet (from segmentation_models_pytorch)
            if attn_path.exists():
                import segmentation_models_pytorch as smp
                self.vessel_attn = smp.Unet(
                    encoder_name="resnet34",
                    decoder_attention_type="scse",
                    in_channels=3,
                    classes=1,
                ).to(self.device)
                state = torch.load(str(attn_path), map_location=self.device, weights_only=False)
                self.vessel_attn.load_state_dict(state["model_state_dict"])
                self.vessel_attn.eval()
                logger.info("Loaded AttentionUNet from %s", attn_path)
            else:
                logger.warning("AttentionUNet checkpoint not found: %s", attn_path)

            # 3. SwinUNet
            if swin_path.exists():
                from app.models.vessel_unet import SwinUNet
                self.vessel_swin = SwinUNet(out_channels=1, pretrained=False).to(self.device)
                state = torch.load(str(swin_path), map_location=self.device, weights_only=False)
                self.vessel_swin.load_state_dict(state["model_state_dict"])
                self.vessel_swin.eval()
                logger.info("Loaded SwinUNet from %s", swin_path)
            else:
                logger.warning("SwinUNet checkpoint not found: %s", swin_path)

            # At least one model must be loaded
            loaded_count = sum(
                1
                for m in [self.vessel_csnet, self.vessel_attn, self.vessel_swin]
                if m is not None
            )
            if loaded_count > 0:
                self.vessel_loaded = True
                self.using_fallback = False
                logger.info(
                    "Vessel ensemble ready: %d/3 models loaded on %s",
                    loaded_count,
                    self.device,
                )
            else:
                logger.warning("No vessel models loaded, using fallback")

        except Exception as exc:
            logger.error("Failed to load vessel models: %s", exc, exc_info=True)
            self.vessel_loaded = False
            self.using_fallback = True

    def _load_grader_and_lesion_models(self) -> None:
        """Load the ConvNeXt grader and the lesion models used to build its channels."""
        grader_path = Path(self.settings.grader_model_path)
        checkpoint_dir = grader_path.parent

        lesion_attention_path = checkpoint_dir / "lesion_attention_unet.pth"
        lesion_unetpp_path = checkpoint_dir / "lesion_unetpp.pth"
        legacy_lesion_path = checkpoint_dir / "lesion_unet.pth"
        lesion_yolo_path = checkpoint_dir / "lesion_yolo.pt"

        # Grader
        try:
            from app.models.grader_cnn import load_grader_model

            if grader_path.exists():
                try:
                    self.grader_model = load_grader_model(grader_path, self.device)
                    self.grader_loaded = True
                    logger.info("Loaded grader model from %s", grader_path)
                except NotImplementedError as nie:
                    logger.warning("Grader checkpoint needs architecture/scripted model: %s", nie)
                    self.grader_model = None
                    self.grader_loaded = False
            else:
                logger.info("Grader checkpoint not found: %s", grader_path)
        except Exception as exc:
            logger.error("Failed to load grader model: %s", exc, exc_info=True)
            self.grader_model = None
            self.grader_loaded = False

        # Lesion models used by the Stage B mask generator:
        # 0.4 Attention U-Net + 0.4 U-Net/UNetPP + 0.2 YOLO.
        try:
            if lesion_attention_path.exists():
                self.lesion_attention_model = self._load_smp_lesion_model(
                    lesion_attention_path, "Attention lesion U-Net"
                )
            else:
                logger.info("Attention lesion checkpoint not found: %s", lesion_attention_path)

            unetpp_source = (
                lesion_unetpp_path if lesion_unetpp_path.exists() else legacy_lesion_path
            )
            if unetpp_source.exists():
                self.lesion_unetpp_model = self._load_smp_lesion_model(
                    unetpp_source, "U-Net/UNetPP lesion model"
                )
            else:
                logger.info("U-Net/UNetPP lesion checkpoint not found: %s", unetpp_source)

            if lesion_yolo_path.exists():
                self.lesion_yolo_model = self._load_yolo_lesion_model(lesion_yolo_path)
            else:
                logger.info("YOLO lesion checkpoint not found: %s", lesion_yolo_path)

            self.lesion_model = self.lesion_unetpp_model or self.lesion_attention_model
        except Exception as exc:
            logger.error("Failed to load lesion models: %s", exc, exc_info=True)
            self.lesion_model = None
            self.lesion_attention_model = None
            self.lesion_unetpp_model = None
            self.lesion_yolo_model = None

    def _load_smp_lesion_model(self, path: Path, label: str) -> torch.nn.Module:
        import segmentation_models_pytorch as smp

        obj = torch.load(str(path), map_location=self.device, weights_only=False)
        if isinstance(obj, torch.nn.Module):
            model = obj.to(self.device)
            model.eval()
            logger.info("Loaded %s nn.Module from %s", label, path)
            return model

        if not isinstance(obj, dict) or ("model_state_dict" not in obj and "state_dict" not in obj):
            raise RuntimeError(f"Unsupported lesion checkpoint format: {path}")

        config = obj.get("config", {})
        out_channels = int(config.get("out_channels", 5))
        model = smp.Unet(
            encoder_name="resnet34",
            in_channels=3,
            classes=out_channels,
            encoder_weights=None,
        )
        state_dict = obj.get("model_state_dict", obj.get("state_dict"))
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        logger.info(
            "Loaded %s from %s (epoch=%s, val_dice=%s)",
            label,
            path,
            obj.get("epoch", "unknown"),
            obj.get("val_dice", "unknown"),
        )
        return model

    def _load_yolo_lesion_model(self, path: Path):
        yolo_config_dir = path.parent / ".ultralytics"
        yolo_config_dir.mkdir(exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir))

        try:
            from ultralytics import YOLO
        except ImportError:
            logger.warning(
                "YOLO lesion checkpoint exists at %s, but ultralytics is not installed; "
                "continuing with the two PyTorch lesion models.",
                path,
            )
            return None

        model = YOLO(str(path))
        logger.info("Loaded YOLO lesion model from %s", path)
        return model

    def _predict_lesions(self, preprocessed_rgb: np.ndarray, tensor_img: torch.Tensor) -> dict:
        """Run the lesion ensemble. Returns dict name->probability map (H,W)."""
        lesion_masks = {}
        mask_names = self.preprocess_config.get("grader", {}).get(
            "mask_channels",
            ["vessel", "MA", "HE", "EX", "OD", "CW"],
        )
        lesion_names = mask_names[1:]
        if (
            self.lesion_attention_model is None
            and self.lesion_unetpp_model is None
            and self.lesion_yolo_model is None
        ):
            return {}
        try:
            expected_shape = (self.settings.model_image_size, self.settings.model_image_size)
            lesion_prob = np.zeros((len(lesion_names), *expected_shape), dtype=np.float32)

            with torch.no_grad():
                if self.lesion_attention_model is not None:
                    lesion_prob += 0.4 * self._lesion_tensor_probs(
                        self.lesion_attention_model,
                        tensor_img,
                        len(lesion_names),
                        expected_shape,
                    )
                if self.lesion_unetpp_model is not None:
                    lesion_prob += 0.4 * self._lesion_tensor_probs(
                        self.lesion_unetpp_model,
                        tensor_img,
                        len(lesion_names),
                        expected_shape,
                    )

            if self.lesion_yolo_model is not None:
                lesion_prob += 0.2 * self._predict_yolo_lesions(
                    preprocessed_rgb,
                    len(lesion_names),
                    expected_shape,
                )

            lesion_prob = np.clip(lesion_prob, 0.0, 1.0)
            for i, name in enumerate(lesion_names):
                lesion_masks[name] = lesion_prob[i]
        except Exception as exc:
            logger.error("Lesion inference failed: %s", exc, exc_info=True)
        return lesion_masks

    def _lesion_tensor_probs(
        self,
        model: torch.nn.Module,
        tensor_img: torch.Tensor,
        expected_channels: int,
        expected_shape: tuple[int, int],
    ) -> np.ndarray:
        out = torch.sigmoid(model(tensor_img)).cpu().numpy()
        chans = out[0] if out.ndim == 4 else out
        if chans.ndim == 2:
            chans = np.expand_dims(chans, 0)
        if chans.shape[0] != expected_channels:
            logger.warning(
                "Lesion model output channels (%d) != expected (%d); ignoring output",
                chans.shape[0],
                expected_channels,
            )
            return np.zeros((expected_channels, *expected_shape), dtype=np.float32)
        return self._resize_channel_stack(chans, expected_shape)

    def _predict_yolo_lesions(
        self,
        preprocessed_rgb: np.ndarray,
        expected_channels: int,
        expected_shape: tuple[int, int],
    ) -> np.ndarray:
        prob_yolo = np.zeros((expected_channels, *expected_shape), dtype=np.float32)
        try:
            result = self.lesion_yolo_model.predict(
                preprocessed_rgb,
                imgsz=self.settings.model_image_size,
                verbose=False,
                retina_masks=True,
                device=0 if self.device.type == "cuda" else "cpu",
            )[0]
            if result.masks is None:
                return prob_yolo
            masks_data = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for idx, cls_id in enumerate(classes):
                if 0 <= cls_id < expected_channels:
                    mask = masks_data[idx]
                    if mask.shape != expected_shape:
                        mask = cv2.resize(
                            mask,
                            expected_shape[::-1],
                            interpolation=cv2.INTER_LINEAR,
                        )
                    prob_yolo[cls_id] = np.maximum(prob_yolo[cls_id], mask.astype(np.float32))
        except Exception as exc:
            logger.error("YOLO lesion inference failed: %s", exc, exc_info=True)
        return np.clip(prob_yolo, 0.0, 1.0)

    def _resize_channel_stack(
        self,
        chans: np.ndarray,
        expected_shape: tuple[int, int],
    ) -> np.ndarray:
        if chans.shape[1:] == expected_shape:
            return chans.astype(np.float32)
        resized = [
            cv2.resize(channel, expected_shape[::-1], interpolation=cv2.INTER_LINEAR)
            for channel in chans
        ]
        return np.stack(resized).astype(np.float32)

    def _grader_predict(
        self,
        preprocessed_rgb: np.ndarray,
        vessel_channel: np.ndarray,
        tensor_img: torch.Tensor,
    ) -> tuple[tuple, float]:
        """Run grader model if loaded, else raise."""
        if not self.grader_loaded or self.grader_model is None:
            raise RuntimeError("No grader model loaded")
        try:
            prepared = Image.fromarray(preprocessed_rgb.astype(np.uint8))
            lesion_masks = self._predict_lesions(preprocessed_rgb, tensor_img)

            grader_input = build_grader_input_array(
                prepared,
                vessel_channel,
                self.preprocess_config,
                lesion_masks,
            )
            arr = grader_input.astype(np.float32)
            tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.grader_model(tensor)

            # Handle tensor outputs and compute per-grade probabilities
            if isinstance(out, torch.Tensor):
                logits = out.cpu().numpy()
                # If spatial logits, average spatial dims
                if logits.ndim > 2:
                    # logits shape (B,C,H,W) or (B,H,W,C)
                    if logits.shape[0] == 1 and logits.shape[1] == 5:
                        probs = softmax(logits[0].mean(axis=(1, 2)))
                    elif logits.shape[0] == 1 and logits.shape[-1] == 5:
                        probs = softmax(logits[0].mean(axis=(0, 1)))
                    else:
                        probs = softmax(np.ravel(logits).astype(float))
                else:
                    probs = softmax(np.asarray(logits).flatten())
            else:
                probs = softmax(np.asarray(out).flatten())

            grade_probs = tuple(float(round(float(p), 3)) for p in probs)
            entropy = predictive_entropy(probs)
            return grade_probs, entropy
        except Exception as exc:
            logger.error("Grader inference failed: %s", exc, exc_info=True)
            raise

    def status(self) -> dict:
        return {
            "vessel_models": {
                "csnet": self.vessel_csnet is not None,
                "attention_unet": self.vessel_attn is not None,
                "swin_unet": self.vessel_swin is not None,
            },
            "vessel_loaded": self.vessel_loaded,
            "grader_loaded": self.grader_loaded,
            "lesion_loaded": self.lesion_model is not None or self.lesion_yolo_model is not None,
            "lesion_models": {
                "attention_unet": self.lesion_attention_model is not None,
                "unetpp": self.lesion_unetpp_model is not None,
                "yolo": self.lesion_yolo_model is not None,
            },
            "using_fallback": self.using_fallback,
            "device": str(self.device),
            "clahe_enabled": True,
        }

    def predict(self, image: Image.Image) -> Prediction:
        if self.vessel_loaded:
            return self._ensemble_predict(image)
        return self._fallback_predict(image)

    def _preprocess_for_vessel(self, image: Image.Image) -> tuple[np.ndarray, torch.Tensor]:
        """Preprocess image exactly as in the training notebook.

        Pipeline: crop_roi → apply_clahe → resize 512 → normalize → tensor
        """
        size = self.settings.model_image_size

        # Convert to numpy RGB
        rgb = np.asarray(image.convert("RGB"))

        # 1. Crop ROI (remove black borders)
        rgb = crop_roi(rgb)

        # 2. Apply CLAHE in LAB space
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        enhanced = cv2.merge((enhanced_l, a_channel, b_channel))
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        # 3. Resize to model input size
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)

        # 4. Normalize with ImageNet stats
        tensor = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        tensor = (tensor - mean) / std

        # 5. Convert to PyTorch tensor (B, C, H, W)
        tensor = torch.tensor(tensor).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        return rgb, tensor

    def _ensemble_predict(self, image: Image.Image) -> Prediction:
        """Run the 3-model vessel ensemble — the real prediction pipeline."""
        preprocessed_rgb, tensor_img = self._preprocess_for_vessel(image)
        size = self.settings.model_image_size
        prepared_image = Image.fromarray(preprocessed_rgb.astype(np.uint8))

        with torch.no_grad():
            probs = []
            weights = []

            # CSNet (weight 0.4)
            if self.vessel_csnet is not None:
                prob_csnet = torch.sigmoid(self.vessel_csnet(tensor_img)).cpu().numpy()[0, 0]
                probs.append(prob_csnet)
                weights.append(0.4)

            # AttentionUNet (weight 0.4)
            if self.vessel_attn is not None:
                prob_attn = torch.sigmoid(self.vessel_attn(tensor_img)).cpu().numpy()[0, 0]
                probs.append(prob_attn)
                weights.append(0.4)

            # SwinUNet (weight 0.2)
            if self.vessel_swin is not None:
                prob_swin = torch.sigmoid(self.vessel_swin(tensor_img)).cpu().numpy()[0, 0]
                probs.append(prob_swin)
                weights.append(0.2)

        # Weighted ensemble
        if probs:
            total_weight = sum(weights)
            vessel_prob = sum(w * p for w, p in zip(weights, probs)) / total_weight
        else:
            vessel_prob = np.zeros((size, size), dtype=np.float32)

        # Threshold to get binary mask
        vessel_threshold = float(self.preprocess_config.get("vessel", {}).get("threshold", 0.5))
        raw_mask = (vessel_prob > vessel_threshold).astype(np.uint8) * 255
        clean_mask = clean_binary_mask(raw_mask)

        # Prob map for heatmap visualization
        prob_map = np.clip(vessel_prob, 0.0, 1.0)

        # Try grader CNN with the soft vessel probability map used in training.
        try:
            grade_probs, entropy = self._grader_predict(preprocessed_rgb, vessel_prob, tensor_img)
        except Exception as exc:
            logger.warning("Using deterministic grade fallback because grader failed: %s", exc)
            grade_probs, entropy = self._deterministic_grade(clean_mask)

        return Prediction(
            prepared_image=prepared_image,
            prob_map=prob_map,
            clean_mask=clean_mask,
            grade_probs=grade_probs,
            entropy=entropy,
            mc_dropout_std=0.05,
        )

    def _deterministic_grade(self, clean_mask: np.ndarray) -> tuple[tuple, float]:
        """Deterministic DR grading based on vessel density (fallback for grader CNN)."""
        density = float((clean_mask > 0).mean())
        severity_signal = np.clip((density - 0.08) / 0.12, 0, 1)
        logits = np.array([
            2.5 - 2.8 * severity_signal,
            1.2 - 0.3 * abs(severity_signal - 0.25),
            0.4 + 1.2 * severity_signal,
            -0.2 + 1.6 * max(0, severity_signal - 0.55),
            -0.6 + 1.8 * max(0, severity_signal - 0.75),
        ])
        probs = softmax(logits, temperature=1.2)
        grade_probs = tuple(float(round(p, 3)) for p in probs)
        entropy = predictive_entropy(probs)
        return grade_probs, entropy

    def _fallback_predict(self, image: Image.Image) -> Prediction:
        """Green-channel fallback when no models are loaded."""
        size = self.settings.model_image_size
        rgb = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS))
        prepared_image = Image.fromarray(rgb.astype(np.uint8))
        green = rgb[:, :, 1]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        vessel_source = clahe.apply(green)

        inverted = 255 - vessel_source
        fov = (green > 12).astype(np.uint8)
        smoothed = cv2.GaussianBlur(inverted, (0, 0), 1.2)
        vessel_threshold = float(self.preprocess_config.get("vessel", {}).get("threshold", 0.5))
        percentile = 50 + min(max(vessel_threshold, 0.0), 1.0) * 66
        threshold = np.percentile(smoothed[fov > 0], percentile) if np.any(fov) else 180
        raw_mask = ((smoothed > threshold) & (fov > 0)).astype(np.uint8) * 255
        clean_mask = clean_binary_mask(raw_mask)
        prob_map = np.clip(smoothed.astype(np.float32) / 255.0, 0.0, 1.0) * fov

        grade_probs, entropy = self._deterministic_grade(clean_mask)

        return Prediction(
            prepared_image=prepared_image,
            prob_map=prob_map,
            clean_mask=clean_mask,
            grade_probs=grade_probs,
            entropy=entropy,
            mc_dropout_std=0.05,
        )


@lru_cache
def get_model_registry() -> ModelRegistry:
    return ModelRegistry()
