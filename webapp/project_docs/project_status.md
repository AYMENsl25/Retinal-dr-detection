# RetinaScope-AI — Project Status & Model Placement Guide

## ✅ Current Status

The project scaffold is **already built and mostly complete**. The code lives at:

```
C:\Users\slima\Downloads\WEB_APP retino\retinascope-ai\
```

The app has:
- **Backend** — FastAPI server (`backend/app/`)
- **Frontend** — Next.js UI (`frontend/`)
- **Docker Compose** — for containerized deployment
- **LLM integration** — Gemini Flash-Lite (via `.env`)
- **Inference engine** — model registry, preprocessing, post-processing, visualizer

---

## 📦 Where to Put Your Model Files

> [!IMPORTANT]
> Your models **already exist** in the raw `model/` folder but need to be in the `backend/checkpoints/` folder for the app to load them.

### ✅ Good news — they are already there!

Checking `backend/checkpoints/`, the models are already copied:

| File | Size | Status |
|---|---|---|
| `grader_cnn.pth` | ~1 GB | ✅ Present |
| `lesion_unet.pth` | ~280 MB | ✅ Present |
| `vessel_unet.pth` | ~649 MB | ✅ Present |
| `preprocess.json` | 1.4 KB | ✅ Present |

### The canonical checkpoint location is:

```
C:\Users\slima\Downloads\WEB_APP retino\retinascope-ai\backend\checkpoints\
├── vessel_unet.pth       ← Vessel segmentation U-Net
├── lesion_unet.pth       ← Lesion segmentation U-Net  
├── grader_cnn.pth        ← DR grading CNN (0–4 scale)
└── preprocess.json       ← Preprocessing config (sizes, normalization)
```

> [!NOTE]
> Your raw originals also sit in `C:\Users\slima\Downloads\WEB_APP retino\model\grading.pt` — this is a **separate copy**, not used by the app directly.

---

## 🗺️ Full Architecture Plan

```
retinascope-ai/
├── backend/
│   ├── checkpoints/          ← 🔴 YOUR MODEL FILES GO HERE
│   │   ├── vessel_unet.pth
│   │   ├── lesion_unet.pth
│   │   ├── grader_cnn.pth
│   │   └── preprocess.json   ← preprocessing params
│   └── app/
│       ├── models/           ← 🔴 Architecture class definitions
│       │   ├── vessel_unet.py
│       │   ├── lesion_unet.py
│       │   ├── grader_cnn.py
│       │   └── registry.py   ← loads & caches models on startup
│       ├── core/
│       │   ├── preprocessing.py
│       │   ├── postprocessing.py
│       │   └── visualizer.py
│       └── llm/              ← Gemini/Claude API integration
│
├── frontend/                 ← Next.js UI
│   └── src/app/analyze/      ← Main analysis page
│
└── .env                      ← API keys & LLM config
```

---

## 🚀 What the Plan Builds (12 Phases)

| Phase | What | Status |
|---|---|---|
| 0 — Repo bootstrap | Monorepo, Docker, `.env` | ✅ Done |
| 1 — Checkpoint validation | Prove models load correctly | ⚠️ Needs running |
| 2 — Inference engine | preprocessing → model → 4-panel output | ✅ Built |
| 3 — FastAPI service | `/analyze` endpoint | ✅ Built |
| 4 — Calibration & uncertainty | Temperature scaling, MC-Dropout | ✅ Built |
| 5 — LLM-1 Clinical Explainer | Structured clinical narrative | ✅ Built |
| 6 — LLM-2 Vascular Analyst | Vision LLM on 4-panel image | ✅ Built |
| 7 — Frontend skeleton | Upload → viewer → grade card | ✅ Built |
| 8 — Reports & chat UI | LLM panels + streaming chat | ✅ Built |
| 9 — Persistence & auth | Postgres + MinIO + NextAuth | 🔲 Optional |
| 10 — PDF export | One-click report export | ✅ Built |
| 11 — Testing & hardening | Unit + integration tests | 🔲 Optional |
| 12 — Deployment | Docker Compose prod | 🔲 Optional |

---

## ⚡ How to Run the App Right Now

### 1. Start the Backend
```powershell
cd "C:\Users\slima\Downloads\WEB_APP retino\retinascope-ai\backend"
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

### 2. Start the Frontend (separate terminal)
```powershell
cd "C:\Users\slima\Downloads\WEB_APP retino\retinascope-ai\frontend"
pnpm dev
```

### 3. Open the app
```
http://localhost:3000/analyze
```

### 4. Verify backend health
```
http://127.0.0.1:8000/api/v1/health
```

---

## 🔧 If You Replace a Model

Use the **same filename** if architecture didn't change:

```
backend\checkpoints\vessel_unet.pth
backend\checkpoints\lesion_unet.pth
backend\checkpoints\grader_cnn.pth
```

If the **architecture changed**, also update the matching Python file:

```
backend\app\models\vessel_unet.py
backend\app\models\lesion_unet.py
backend\app\models\grader_cnn.py
```

Then validate:
```powershell
cd "C:\Users\slima\Downloads\WEB_APP retino\retinascope-ai"
python scripts\validate_checkpoints.py
```

---

## 🔑 LLM Configuration (`.env`)

Located at:
```
C:\Users\slima\Downloads\WEB_APP retino\retinascope-ai\.env
```

Current setup uses **Gemini Flash-Lite** for both chat and vision reports.

---

## ❓ Open Questions

> [!IMPORTANT]
> Before running Phase 1 (checkpoint validation), confirm:
> 1. **Do the model architecture classes** in `backend/app/models/vessel_unet.py`, `lesion_unet.py`, and `grader_cnn.py` **match exactly** how you trained the models? If not, they need to be updated with your actual architecture code.
> 2. **Does `preprocess.json`** in `backend/checkpoints/` reflect the correct input sizes, mean/std normalization, and CLAHE settings from training?
> 3. Do you have a **Gemini API key** set in `.env`? The LLM features won't work without it.
