# 🔧 RetinaScope-AI v2 — Implementation Plan

## Problem Summary

Two images from the user clearly show the issue:
- **Image 1** (from notebook): Clean, detailed vessel segmentation with beautiful vessel structure on dark background
- **Image 2** (from webapp): Panel 3 "Binary Vessel Mask" shows white blobs/artifacts instead of real vessel segmentation

### Root Cause
The vessel model file `vessel_unet.py` is a **placeholder stub** that raises `NotImplementedError`. The `ModelRegistry.predict()` always falls back to `_fallback_predict()` which uses a crude green-channel inversion + thresholding approach — that's what produces the blobs.

---

## What Needs to Change

### Part 1: Fix Vessel Segmentation (Critical)

The notebook (`dr-grading-dataset-preparation.ipynb`) reveals the **exact architecture and preprocessing**:

#### Model Files Available in Project Root
| File | Size | Architecture | Purpose |
|------|------|--------------|---------|
| `csnet_best.pt` | 375 MB | CSNet (custom CS-Net with Channel-Spatial Attention) | Vessel segmentation |
| `attentionUNet_best_model.pt` | 560 MB | `smp.Unet(encoder_name='resnet34', decoder_attention_type='scse')` | Vessel segmentation |
| `swin_unet_best.pt` | 125 MB | SwinUNet (Swin Transformer + U-Net decoder) | Vessel segmentation |

#### Ensemble Formula
```python
vessel_prob = 0.4 * csnet + 0.4 * attn_unet + 0.2 * swin_unet
```

#### Preprocessing Pipeline (MUST match training)
1. `crop_roi(img)` — removes black borders
2. `apply_clahe(img)` — CLAHE in LAB color space
3. `cv2.resize(img, (512, 512))`
4. Normalize: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
5. Convert to tensor: `(H,W,C) → (C,H,W)`, add batch dim

#### Changes Required
1. **Copy model weights** from project root to `backend/checkpoints/`
2. **Create `vessel_unet.py`** with the 3 model architectures (CSNet, SwinUNet from notebook; AttentionUNet from `segmentation_models_pytorch`)
3. **Update `registry.py`** to load & run the 3-model ensemble instead of fallback
4. **Update preprocessing** to enable CLAHE and add `crop_roi()`
5. **Add `segmentation-models-pytorch`** and **`timm`** to dependencies

### Part 2: Add Gemini as Primary LLM

Current `LLM_PROVIDER` only supports a single provider. User wants:
1. **Gemini** → primary (free, has API key)
2. **Groq** → backup (free, has API key)  
3. **Deterministic fallback** → last resort

#### Changes Required
1. **Add Gemini provider** to `config.py` and `client.py`
2. **Implement cascading fallback**: try Gemini → try Groq → use deterministic
3. **Update `.env`** with Gemini API key and new provider config

---

## Implementation Steps

### Step 1: Add Dependencies
- Add `segmentation-models-pytorch` and `timm` to `pyproject.toml`

### Step 2: Implement Model Architectures
- Write CSNet and SwinUNet classes in `vessel_unet.py` (from notebook)
- These are needed to load the saved `.pt` checkpoints

### Step 3: Copy Model Weights
- Copy `csnet_best.pt`, `attentionUNet_best_model.pt`, `swin_unet_best.pt` to `backend/checkpoints/`

### Step 4: Rewrite ModelRegistry
- Load all 3 vessel models at startup
- Run ensemble inference with proper CLAHE preprocessing
- Keep deterministic grading fallback (since grader_cnn.py is also a placeholder)

### Step 5: Add Gemini LLM Support
- Add Gemini API settings to config
- Implement `_gemini_json()` method in LLM client
- Implement cascading provider logic: try primary → try fallback → deterministic

### Step 6: Update Environment
- Add `GEMINI_API_KEY` to `.env`
- Set LLM provider chain configuration

> [!IMPORTANT]
> This is a significant set of changes touching the core ML pipeline and LLM client. I'll proceed step by step.
