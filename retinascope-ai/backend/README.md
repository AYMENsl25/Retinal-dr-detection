# Backend

FastAPI service for RetinaScope-AI.

## Model Files

Place files in `backend/checkpoints/`:

```text
vessel_unet.pth
lesion_unet.pth
grader_cnn.pth
preprocess.json
```

The service currently uses a deterministic mock pipeline unless real model classes and checkpoints are available.

## API

- `GET /api/v1/health`
- `POST /api/v1/analyze`
- `POST /api/v1/chat`

## LLM Keys

Use the root `.env.example` as the template. The chosen free test stack is:

- `GROQ_API_KEY` for `llama-3.3-70b-versatile`
- `GEMINI_API_KEY` for `gemini-2.5-flash`
- `OPENROUTER_API_KEY` optional fallback for `openrouter/free`

