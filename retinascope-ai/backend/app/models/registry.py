from dataclasses import dataclass
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from app.config import settings
from app.core.metrics import compute_biomarkers
from app.core.postprocessing import (
    binary_mask,
    compute_mock_vessel_probability,
    grade_from_biomarkers,
    grade_probabilities,
)
from app.core.preprocessing import resize_for_model
from app.core.visualizer import build_panel_data_urls
from app.schemas.inference import BaseAnalyzeResult


@dataclass(frozen=True)
class RegistryStatus:
    runtime: str
    checkpoints: dict[str, bool]


class ModelRegistry:
    def __init__(self):
        self._loaded = False
        self._load_error: str | None = None
        self._device = None
        self._vessel_model = None
        self._lesion_model = None
        self._grader_model = None
        self._preprocess = self._load_preprocess()

    def status(self) -> RegistryStatus:
        checkpoint_status = {
            "vessel_unet": settings.checkpoint_file(settings.vessel_checkpoint).exists(),
            "lesion_unet": settings.checkpoint_file(settings.lesion_checkpoint).exists(),
            "grader_cnn": settings.checkpoint_file(settings.grader_checkpoint).exists(),
            "preprocess": settings.checkpoint_file(settings.preprocess_config).exists(),
        }
        runtime = "mock"
        if all(checkpoint_status.values()):
            runtime = "real-ready" if self._load_error is None else f"real-load-failed: {self._load_error}"
        if self._loaded:
            runtime = "real-loaded"
        return RegistryStatus(runtime=runtime, checkpoints=checkpoint_status)

    def analyze(self, image: Image.Image) -> BaseAnalyzeResult:
        if self._can_try_real_inference():
            try:
                return self._analyze_real(image)
            except Exception as exc:
                self._load_error = str(exc)

        return self._analyze_mock(image)

    def _can_try_real_inference(self) -> bool:
        return all(
            [
                settings.checkpoint_file(settings.vessel_checkpoint).exists(),
                settings.checkpoint_file(settings.lesion_checkpoint).exists(),
                settings.checkpoint_file(settings.grader_checkpoint).exists(),
                settings.checkpoint_file(settings.preprocess_config).exists(),
            ]
        )

    def _load_preprocess(self) -> dict:
        path = settings.checkpoint_file(settings.preprocess_config)
        if not path.exists():
            root_path = settings.resolved_path(settings.preprocess_config)
            path = root_path if root_path.exists() else path
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _checkpoint_state_dict(self, path: Path, torch_module):
        checkpoint = torch_module.load(path, map_location=self._device, weights_only=False)
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    checkpoint = value
                    break
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Checkpoint does not contain a state dict: {path}")
        return {
            key.removeprefix("module."): value
            for key, value in checkpoint.items()
            if hasattr(value, "shape")
        }

    def _load_models(self):
        if self._loaded:
            return

        import torch

        from app.models.grader_cnn import build_grader_model
        from app.models.lesion_unet import build_lesion_model
        from app.models.vessel_unet import build_vessel_model

        if settings.model_device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(settings.model_device)

        if not self._preprocess:
            self._preprocess = self._load_preprocess()

        image_size = int(self._preprocess.get("input_size", [512, 512])[0])

        self._vessel_model = build_vessel_model(img_size=image_size, pretrained=False).to(self._device)
        self._vessel_model.load_state_dict(
            self._checkpoint_state_dict(settings.checkpoint_file(settings.vessel_checkpoint), torch)
        )
        self._vessel_model.eval()

        self._lesion_model = build_lesion_model(encoder_weights=None).to(self._device)
        self._lesion_model.load_state_dict(
            self._checkpoint_state_dict(settings.checkpoint_file(settings.lesion_checkpoint), torch)
        )
        self._lesion_model.eval()

        self._grader_model = build_grader_model(num_classes=5, pretrained=False).to(self._device)
        self._grader_model.load_state_dict(
            self._checkpoint_state_dict(settings.checkpoint_file(settings.grader_checkpoint), torch)
        )
        self._grader_model.eval()

        self._loaded = True
        self._load_error = None

    def _preprocess_rgb_tensor(self, image: Image.Image):
        import torch

        image_size = int(self._preprocess.get("input_size", [512, 512])[0])
        resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()
        mean = torch.tensor(self._preprocess.get("normalize_mean", [0.485, 0.456, 0.406])).view(3, 1, 1)
        std = torch.tensor(self._preprocess.get("normalize_std", [0.229, 0.224, 0.225])).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return resized, tensor.unsqueeze(0).to(self._device)

    def _analyze_real(self, image: Image.Image) -> BaseAnalyzeResult:
        import torch

        self._load_models()
        resized, rgb_tensor = self._preprocess_rgb_tensor(image)

        vessel_threshold = float(self._preprocess.get("vessel", {}).get("threshold", 0.5))
        lesion_threshold = float(self._preprocess.get("lesion", {}).get("threshold", 0.3))

        with torch.no_grad():
            vessel_logits = self._vessel_model(rgb_tensor)
            vessel_prob_t = torch.sigmoid(vessel_logits)[0, 0].detach().cpu()
            vessel_prob = vessel_prob_t.numpy().astype(np.float32)
            vessel_mask = (vessel_prob > vessel_threshold).astype(np.uint8)

            lesion_logits = self._lesion_model(rgb_tensor)
            lesion_prob_t = torch.sigmoid(lesion_logits)[0].detach().cpu()
            lesion_mask_t = (lesion_prob_t > lesion_threshold).float()

            vessel_mask_t = torch.from_numpy(vessel_mask).float().unsqueeze(0)
            mask_stack = torch.cat([vessel_mask_t, lesion_mask_t], dim=0)
            mask_stack = (mask_stack - 0.5) / 0.5

            grader_input = torch.cat([rgb_tensor[0].detach().cpu(), mask_stack], dim=0)
            grader_logits = self._grader_model(grader_input.unsqueeze(0).to(self._device))
            grade_probs_np = torch.softmax(grader_logits, dim=1)[0].detach().cpu().numpy()

        grade_probs = [float(round(value, 6)) for value in grade_probs_np]
        grade = int(np.argmax(grade_probs_np))
        calibrated_confidence = grade_probs[grade]
        next_grade = min(grade + 1, 4)
        denom = grade_probs[grade] + grade_probs[next_grade]
        closeness = 0.0 if grade == 4 or denom == 0 else grade_probs[next_grade] / denom
        entropy = float(-np.sum([p * np.log(p) for p in grade_probs if p > 0]))

        biomarkers = compute_biomarkers(vessel_mask, vessel_prob)

        if entropy >= 0.6 or grade >= 3:
            decision_flag = "REFER_SPECIALIST"
        elif entropy >= 0.3 or closeness > 0.35:
            decision_flag = "MEDIUM_REFER_RECOMMENDED"
        else:
            decision_flag = "HIGH_CONFIDENCE"

        return BaseAnalyzeResult(
            case_id=str(uuid4()),
            panels=build_panel_data_urls(resized, vessel_prob, vessel_mask),
            grade=grade,
            grade_probs=grade_probs,
            calibrated_confidence=round(calibrated_confidence, 6),
            closeness_to_next_grade=round(float(closeness), 6),
            uncertainty={"entropy": round(entropy, 6), "mc_dropout_std": 0.0},
            biomarkers=biomarkers,
            decision_flag=decision_flag,
        )

    def _analyze_mock(self, image: Image.Image) -> BaseAnalyzeResult:
        resized = resize_for_model(image)
        probability = compute_mock_vessel_probability(resized)
        mask = binary_mask(probability)
        biomarkers = compute_biomarkers(mask, probability)
        grade = grade_from_biomarkers(biomarkers["vessel_density"], biomarkers["tortuosity"])
        probs = grade_probabilities(grade)

        calibrated_confidence = probs[grade]
        next_grade = min(grade + 1, 4)
        denom = probs[grade] + probs[next_grade]
        closeness = 0.0 if grade == 4 else probs[next_grade] / denom
        entropy = float(-np.sum([p * np.log(p) for p in probs if p > 0]))

        if entropy >= 0.6 or grade >= 3:
            decision_flag = "REFER_SPECIALIST"
        elif entropy >= 0.3 or closeness > 0.35:
            decision_flag = "MEDIUM_REFER_RECOMMENDED"
        else:
            decision_flag = "HIGH_CONFIDENCE"

        return BaseAnalyzeResult(
            case_id=str(uuid4()),
            panels=build_panel_data_urls(resized, probability, mask),
            grade=grade,
            grade_probs=probs,
            calibrated_confidence=round(calibrated_confidence, 6),
            closeness_to_next_grade=round(float(closeness), 6),
            uncertainty={"entropy": round(entropy, 6), "mc_dropout_std": 0.08},
            biomarkers=biomarkers,
            decision_flag=decision_flag,
        )


model_registry = ModelRegistry()
