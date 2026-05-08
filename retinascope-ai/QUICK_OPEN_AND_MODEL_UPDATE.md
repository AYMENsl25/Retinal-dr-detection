# Quick Open And Model Update Guide

## Open The App Again

Open PowerShell 1 for the backend:

```powershell
cd "C:\Users\slima\Downloads\WEB_APP retino\retinascope-ai\backend"
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

If `.venv` does not exist yet:

```powershell
cd "C:\Users\slima\Downloads\WEB_APP retino\retinascope-ai\backend"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,ml]"
```

Open PowerShell 2 for the frontend:

```powershell
cd "C:\Users\slima\Downloads\WEB_APP retino\retinascope-ai\frontend"
pnpm dev
```

Open:

```text
http://localhost:3000/analyze
```

## Check The Backend

Open:

```text
http://127.0.0.1:8000/api/v1/health
```

If the backend is healthy, you should see `status: ok`.

## Current LLM Setup

The app now uses Gemini Flash-Lite for chat and the vision report:

```text
VISION_LLM_MODEL=gemini-3.1-flash-lite
CHAT_LLM_MODEL=gemini-3.1-flash-lite
```

These values live in:

```text
C:\Users\slima\Downloads\WEB_APP retino\retinascope-ai\.env
```

After changing `.env`, restart the backend.

## Export PDF

Run an analysis first. Then click `Export PDF` in the header.

Your browser print dialog will open. Choose:

```text
Destination -> Save as PDF
```

## If You Replace A Model With A Better One

Use the same filename if the architecture did not change:

```text
backend\checkpoints\vessel_unet.pth
backend\checkpoints\lesion_unet.pth
backend\checkpoints\grader_cnn.pth
```

Then restart the backend.

If the architecture changed, update the matching Python file:

```text
backend\app\models\vessel_unet.py
backend\app\models\lesion_unet.py
backend\app\models\grader_cnn.py
```

If preprocessing changed, update:

```text
backend\checkpoints\preprocess.json
```

Examples:

- New image size -> update `input_size`
- Different threshold -> update `vessel.threshold` or `lesion.threshold`
- Different grader input channels -> update `grader.input_channels` and `registry.py`
- Different normalization -> update `normalize_mean` and `normalize_std`

Always run:

```powershell
cd "C:\Users\slima\Downloads\WEB_APP retino\retinascope-ai"
python scripts\validate_checkpoints.py
```

Then restart:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

