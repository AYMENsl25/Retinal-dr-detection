# RetinaScope-AI v2

This is the v2 build directory created from the current RetinaScope-AI frontend and the `PROJECT_PLAN_v2.md` architecture.

The structure is now:

```text
retinascope-ai-v2/
  frontend/                 # copied from the current Next.js design
  backend/                  # FastAPI + model + vessel damage scaffold
  docs/                     # setup notes for LLMs, models, and vessel damage
  notebooks/                # validation notebooks go here
  .env.example              # root backend/API provider configuration
```

## Where Your Models Go

Put your trained PyTorch checkpoints here:

```text
backend/checkpoints/vessel_unet.pth
backend/checkpoints/grader_cnn.pth
backend/checkpoints/preprocess.json
```

The unified preprocessing JSON now says:

```text
CLAHE: enabled, matching the Stage B generator notebook
Vessel model: 3-model ensemble, 3-channel RGB input, threshold 0.5
Grader model: ConvNeXtBase9Channel, 9-channel input
```

Important: the grader checkpoint expects RGB plus six mask channels:

```text
RGB + vessel + MA + HE + EX + OD + CW
```

The v2 webapp now generates the vessel channel plus the five lesion channels. The lesion channels use the Stage B training ensemble: Attention U-Net, U-Net/UNetPP, and YOLO.

I left architecture stubs in:

```text
backend/app/models/vessel_unet.py
backend/app/models/grader_cnn.py
```

When you share the model architecture details, those are the files to fill in before loading the `.pth` weights.

## Where API Keys Go

Copy the root env file:

```powershell
Copy-Item .env.example .env
```

Then replace the `replace_me_*` values in:

```text
retinascope-ai-v2/.env
```

For the frontend, copy:

```powershell
Copy-Item frontend/.env.example frontend/.env.local
```

The frontend starts in mock mode. Set `NEXT_PUBLIC_USE_MOCK_PIPELINE=false` after the backend is running and your model path is ready.

## Recommended Free LLM Setup

Default choice for this repo: `LLM_PROVIDER=ollama`.

That is the safest free setup because it runs locally, needs no API key, and keeps medical images off hosted services. Use:

```text
OLLAMA_TEXT_MODEL=llama3.1:8b
OLLAMA_VISION_MODEL=llama3.2-vision:11b
```

Good hosted free-tier option:

```text
LLM_PROVIDER=openrouter
OPENROUTER_TEXT_MODEL=openrouter/free
OPENROUTER_VISION_MODEL=openrouter/free
OPENROUTER_API_KEY=replace_me_openrouter_key
```

Groq is excellent for fast free-tier text reports/chat, but keep vessel image annotation on Ollama/OpenRouter/Claude because Groq is mainly useful here for text.

See `docs/LLM_AND_MODEL_SETUP.md` for the full comparison.

## Run The Web App

Terminal 1, backend:

```powershell
cd C:\Users\slima\Downloads\medical-retina-scope-ai\retinascope-ai-v2\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2, frontend:

```powershell
cd C:\Users\slima\Downloads\medical-retina-scope-ai\retinascope-ai-v2\frontend
pnpm install
pnpm dev
```

Open:

```text
http://localhost:3000
```

## Version

The new build is marked as `2.0.0-dev` in `VERSION`, the frontend package, and backend settings.

## Vessel Damage Detection Recommendation

For damaged vessels, do not rely on the LLM alone. The best v2 path is:

1. Segment vessels with your U-Net.
2. Skeletonize the binary vessel mask.
3. Detect candidate damage regions using graph/connectivity features: endpoints, broken components, tortuosity, local vessel density drops, and caliber irregularity from distance transforms.
4. Send only the clean mask plus candidate regions/biomarkers to a vision LLM for explanation and final JSON cleanup.
5. Draw red ellipses from validated coordinates only.

This scaffold starts that approach in `backend/app/core/vessel_analysis.py`.
