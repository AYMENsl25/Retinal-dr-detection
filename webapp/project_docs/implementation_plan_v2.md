# 🔧 RetinaScope-AI v2 — Grading & Lesion Model Pipeline Implementation Plan

## Problem Summary
The diabetic retinopathy (DR) grading system in the backend currently outputs only Grade 0 results. 

### Root Causes
1. **Grader Model Loading Failure**: The model loading helper (`grader_cnn.py`) expects either a serialized PyTorch `nn.Module` or a TorchScript file. The actual checkpoint files (`grade_best_model.pt` and `grader_cnn.pth`) are standard dictionary checkpoints containing a `model_state_dict` without the code definition. As a result, loading fails with a `NotImplementedError`, and the application silently falls back to a crude deterministic grading function that always predicts Grade 0 for typical vessel patterns.
2. **Lesion Model Loading Failure**: Similarly, the lesion segmentation model checkpoint (`lesion_unet.pth` and the newly uploaded lesion models) contains a dictionary checkpoint (`model_state_dict`). The loader in `registry.py` doesn't handle dictionary checkpoints for lesions and fails to load them. Consequently, no lesion masks (MA, HE, EX, OD, CW) are produced, leaving the grading model with empty lesion input channels.
3. **Workspace Cleanup**: There are multiple model checkpoints (`grade_best_model.pt`, `UNetPP_ResNet34_best.pth`, `Attention_UNet_ResNet34_best.pth`, `YOLOV11_best.pt`) directly in the workspace root, which need to be cleaned up, copied to their correct locations, and the workspace organized.

---

## Lesion Model Comparison And Runtime Choice
The user uploaded two new lesion models and asked which one is better:
- **`UNetPP_ResNet34_best.pth`**: Epoch 21, **Validation Dice: 0.4868**
- **`Attention_UNet_ResNet34_best.pth`**: Epoch 79, **Validation Dice: 0.4571**
- **`lesion_unet.pth` (Previous)**: Epoch 57, **Validation Dice: 0.4750**

Through model checkpoint inspection, we found that all three checkpoints actually share the same architecture: a standard U-Net (`smp.Unet(encoder_name="resnet34", in_channels=3, classes=5)`). 

**Recommendation**: The new `UNetPP_ResNet34_best.pth` has the highest Validation Dice score (**0.4868**) among the PyTorch lesion checkpoints, so it remains the best single PyTorch lesion model.

**Runtime choice for grading**: The grading model was trained from the Stage B generator, which used a lesion ensemble:

```python
lesion_prob = 0.4 * attention_unet + 0.4 * unetpp + 0.2 * yolo
```

So the backend should not choose only one lesion model for grading. It should load both PyTorch lesion checkpoints plus YOLO when available, apply sigmoid to PyTorch logits, keep soft probability masks, and feed those soft lesion channels to the 9-channel grader.

---

## Proposed Changes

### 1. ML Pipeline Architecture

#### [MODIFY] [grader_cnn.py](file:///c:/Users/slima/Downloads/medical-retina-scope-ai/retinascope-ai-v2/backend/app/models/grader_cnn.py)
Update `load_grader_model()` to instantiate a 9-channel ConvNeXt-Base model using `timm.create_model('convnext_base', pretrained=False, in_chans=9, num_classes=5)` and load the state dictionary if the checkpoint is a `state_dict` dictionary.

#### [MODIFY] [registry.py](file:///c:/Users/slima/Downloads/medical-retina-scope-ai/retinascope-ai-v2/backend/app/models/registry.py)
Update `_load_grader_and_lesion_models()` to properly load dictionary checkpoints for the lesion models. The two PyTorch lesion checkpoints both match a standard `smp.Unet` with ResNet34 encoder and 5 output classes, so instantiate `smp.Unet(encoder_name="resnet34", in_channels=3, classes=5, encoder_weights=None)` and load each `model_state_dict`. Load YOLO through `ultralytics` and combine the three outputs with the Stage B weights.

---

### 2. Checkpoint Management & File Copying
1. Overwrite `retinascope-ai-v2/backend/checkpoints/grader_cnn.pth` with `grade_best_model.pt` from the root directory.
2. Overwrite `retinascope-ai-v2/backend/checkpoints/lesion_unet.pth` with `UNetPP_ResNet34_best.pth` for legacy compatibility.
3. Copy `UNetPP_ResNet34_best.pth` to `retinascope-ai-v2/backend/checkpoints/lesion_unetpp.pth`.
4. Copy `Attention_UNet_ResNet34_best.pth` to `retinascope-ai-v2/backend/checkpoints/lesion_attention_unet.pth`.
5. Copy `YOLOV11_best.pt` to `retinascope-ai-v2/backend/checkpoints/lesion_yolo.pt`.

---

### 3. Folder Cleanup
Clean up the workspace root directory:
1. Delete temporary/duplicate scripts in the backend directory (e.g., `inspect_models.py`, `inspect_keys.py`, `test_loader.py`, `test_lesion.py`).
2. Move markdown/plan files from the workspace root (if any) or project docs to a dedicated `project_docs/` or `docs/` folder to organize them.
3. Remove the redundant files `grade_best_model.pt`, `UNetPP_ResNet34_best.pth`, `Attention_UNet_ResNet34_best.pth` from the workspace root once they have been copied to the `backend/checkpoints/` directory.

---

## Verification Plan

### Automated Tests
1. Run a script that performs dummy inference using `ModelRegistry` with the loaded 9-channel ConvNeXt-Base grader and the 5-channel ResNet34 U-Net lesion model on a sample image.
2. Verify that the grader path executes without falling back and that the health status reports all lesion models loaded.
3. Run the backend unit tests to ensure no regressions.

### Manual Verification
1. Run the FastAPI backend server (`uvicorn app.main:app`).
2. Verify that the server logs show successful loading of both the grader and lesion models:
   - `Loaded grader model with ConvNeXt-Base architecture from ...`
   - `Loaded lesion model from dictionary checkpoint ...`
