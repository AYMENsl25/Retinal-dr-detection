# RetinaScope-AI

Clinical decision-support scaffold for diabetic retinopathy screening.

This directory was created from the current Next.js design and the two planning documents:

- `PROJECT_PLAN.md`
- `LLM_SELECTION_GUIDE.md`

## Structure

```text
retinascope-ai/
  backend/                 FastAPI API, model registry, LLM layer
    app/
    checkpoints/           Put your .pth files here
  frontend/                Copied Next.js/shadcn UI from the existing app
  docs/                    Implementation notes
  scripts/                 Validation helpers
```

## Where Your Models Go

Put your trained PyTorch checkpoints in:

```text
retinascope-ai/backend/checkpoints/
  vessel_unet.pth
  lesion_unet.pth
  grader_cnn.pth
  preprocess.json
```

The architecture class stubs live in:

```text
retinascope-ai/backend/app/models/
  vessel_unet.py
  lesion_unet.py
  grader_cnn.py
```

Replace those stubs with the exact Python classes used during training, then update `registry.py` if the checkpoint keys differ.

## Free LLM Choice

For the test version I chose:

- Clinical text report: Groq `llama-3.3-70b-versatile`
- Vision vascular report: Google Gemini `gemini-2.5-flash`
- Optional fallback: OpenRouter `openrouter/free`

Required keys:

```text
GROQ_API_KEY=replace-with-your-groq-api-key
GEMINI_API_KEY=replace-with-your-gemini-api-key
OPENROUTER_API_KEY=replace-with-your-openrouter-api-key
```

The backend defaults to mock-safe behavior until valid keys are present. Do not send real patient data to free tiers.

## Start Locally

Backend:

```powershell
cd retinascope-ai\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
uvicorn app.main:app --reload --port 8000
```

Frontend:

```powershell
cd retinascope-ai\frontend
pnpm install
$env:BACKEND_API_URL="http://127.0.0.1:8000"
pnpm dev
```

Open `http://localhost:3000/analyze`.

