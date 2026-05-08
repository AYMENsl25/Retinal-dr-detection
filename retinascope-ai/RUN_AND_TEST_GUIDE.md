# RetinaScope-AI Run And Test Guide

This guide explains how to run the project after you add your model files, API keys, and model architecture code.

Project folder:

```text
C:\Users\slima\Downloads\WEB_APP retino\retinascope-ai
```

## 1. Add Your Model Files

Put your trained PyTorch checkpoints here:

```text
C:\Users\slima\Downloads\WEB_APP retino\retinascope-ai\backend\checkpoints\
```

Required files:

```text
vessel_unet.pth
lesion_unet.pth
grader_cnn.pth
preprocess.json
```

You can create `preprocess.json` from the example file:

```powershell
cd "C:\Users\slima\Downloads\WEB_APP retino\retinascope-ai"
Copy-Item backend\checkpoints\preprocess.example.json backend\checkpoints\preprocess.json
```

Then edit `backend\checkpoints\preprocess.json` with the exact preprocessing used during training:

```json
{
  "input_size": [512, 512],
  "color_space": "RGB",
  "normalize_mean": [0.485, 0.456, 0.406],
  "normalize_std": [0.229, 0.224, 0.225],
  "clahe": {
    "enabled": true,
    "clip_limit": 2.0,
    "tile_grid_size": [8, 8]
  },
  "threshold": 0.5
}
```

## 2. Add Your Model Architecture Code

Paste the exact Python model classes used during training into:

```text
backend\app\models\vessel_unet.py
backend\app\models\lesion_unet.py
backend\app\models\grader_cnn.py
```

After that, update this file so it loads your real checkpoints instead of the mock pipeline:

```text
backend\app\models\registry.py
```

## 3. Add Your LLM API Keys

Open:

```text
C:\Users\slima\Downloads\WEB_APP retino\retinascope-ai\.env
```

Replace these placeholder values:

```text
GROQ_API_KEY=replace-with-your-groq-api-key
GEMINI_API_KEY=replace-with-your-gemini-api-key
OPENROUTER_API_KEY=replace-with-your-openrouter-api-key
```

Recommended free test stack:

```text
Clinical report LLM: Groq llama-3.3-70b-versatile
Vision report LLM:   Gemini gemini-2.5-flash
Fallback LLM:        OpenRouter openrouter/free
```

`OPENROUTER_API_KEY` is optional. Groq and Gemini are the important keys.

Important: use these free LLM tiers only for public, synthetic, or test images. Do not send real patient data to free-tier APIs.

## 4. Run The Backend

Open PowerShell:

```powershell
cd "C:\Users\slima\Downloads\WEB_APP retino\retinascope-ai\backend"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

For real model inference, install the ML dependencies too. This includes PyTorch, torchvision, timm, and segmentation-models-pytorch:

```powershell
pip install -e ".[dev,ml]"
```

Check whether your checkpoint files are in the correct place:

```powershell
python ..\scripts\validate_checkpoints.py
```

Start the backend:

```powershell
uvicorn app.main:app --reload --port 8000
```

Keep this PowerShell window open.

Backend test URLs:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/api/v1/health
```

If `/api/v1/health` works, the backend is running.

## 5. Run The Frontend

Open a second PowerShell window:

```powershell
cd "C:\Users\slima\Downloads\WEB_APP retino\retinascope-ai\frontend"
```

If you already have `pnpm`:

```powershell
pnpm install
```

If `pnpm` is not installed, install Node.js LTS first, then run:

```powershell
corepack enable
corepack prepare pnpm@latest --activate
pnpm install
```

Create the frontend environment file:

```powershell
Copy-Item .env.local.example .env.local
```

Open `.env.local` and make sure it contains:

```text
BACKEND_API_URL=http://127.0.0.1:8000
```

Start the frontend:

```powershell
pnpm dev
```

Open:

```text
http://localhost:3000/analyze
```

Upload a fundus image and click `Run analysis`.

## 6. How To Test The App

### Test 1: Backend Health

Open:

```text
http://127.0.0.1:8000/api/v1/health
```

Expected result:

```json
{
  "status": "ok",
  "model_runtime": "mock",
  "checkpoints": {
    "vessel_unet": true,
    "lesion_unet": true,
    "grader_cnn": true,
    "preprocess": true
  }
}
```

If the checkpoint values are `false`, your files are missing or named differently.

### Test 2: API Documentation

Open:

```text
http://127.0.0.1:8000/docs
```

Use Swagger UI to test:

```text
POST /api/v1/analyze
```

Upload a fundus image and execute the request.

### Test 3: Frontend Upload Flow

Open:

```text
http://localhost:3000/analyze
```

Upload a fundus image.

Expected result:

- Four-panel viewer appears.
- Grade card appears.
- Clinical report appears.
- Vascular report appears.
- Consultation chat appears.

### Test 4: Chat

After analysis, ask:

```text
What is the follow-up window?
```

Expected result:

The chat should answer using the current case context.

## 7. Current Mock Behavior

Until your real PyTorch loading is wired into `backend\app\models\registry.py`, the backend uses a mock-safe image pipeline.

That means:

- Upload works.
- Four-panel output works.
- API contract works.
- LLM wiring is in place.
- Real `.pth` inference is not active yet.

The next implementation step is replacing the mock inference in:

```text
backend\app\models\registry.py
```

with real PyTorch model loading and prediction.

## 8. Common Problems

### `pnpm` is not recognized

Install Node.js LTS, then run:

```powershell
corepack enable
corepack prepare pnpm@latest --activate
```

### Backend says checkpoint is missing

Run:

```powershell
python ..\scripts\validate_checkpoints.py
```

Make sure the files are named exactly:

```text
vessel_unet.pth
lesion_unet.pth
grader_cnn.pth
preprocess.json
```

### Frontend cannot reach backend

Check `frontend\.env.local`:

```text
BACKEND_API_URL=http://127.0.0.1:8000
```

Then restart `pnpm dev`.

### LLM reports or chat are still mock reports

Check `.env`:

```text
LLM_MODE=auto
GEMINI_API_KEY=your-real-key
VISION_LLM_MODEL=gemini-3.1-flash-lite
CHAT_LLM_MODEL=gemini-3.1-flash-lite
```

Restart the backend after changing `.env`.

### Export PDF is disabled

Run an analysis first. The `Export PDF` button becomes active after a case result exists.
