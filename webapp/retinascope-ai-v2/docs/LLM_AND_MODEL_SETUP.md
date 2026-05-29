# LLM And Model Setup

## Chosen Default

I set the repo default to:

```text
LLM_PROVIDER=ollama
OLLAMA_TEXT_MODEL=llama3.1:8b
OLLAMA_VISION_MODEL=llama3.2-vision:11b
```

Why: it is free, local, and does not send retinal images to an outside API. That matters for a medical-imaging workflow.

## Free LLM Options

| Provider | Use it for | API key | Notes |
|---|---|---|---|
| Ollama | Clinical report, chat, local vision review | No | Best privacy. Needs local RAM/GPU/CPU. |
| OpenRouter | Hosted free text/vision models | Yes | Use `openrouter/free` or specific `:free` models. Availability can change. |
| Groq | Very fast hosted text reports/chat | Yes | Great for LLM-1 and chat. Keep vision on Ollama/OpenRouter/Claude. |
| Hugging Face local/downloaded models | Research and self-hosting | Usually no for local | Good for Qwen/Llama open weights if your machine can run them. |

## Recommended Model Split

For free/local development:

```text
LLM_PROVIDER=ollama
OLLAMA_TEXT_MODEL=llama3.1:8b
OLLAMA_VISION_MODEL=llama3.2-vision:11b
```

For stronger open vision on your own hardware, evaluate Qwen vision models:

```text
Qwen/Qwen2.5-VL-7B-Instruct
Qwen/Qwen3-VL-32B-Instruct
```

For hosted free experiments:

```text
LLM_PROVIDER=openrouter
OPENROUTER_TEXT_MODEL=openrouter/free
OPENROUTER_VISION_MODEL=openrouter/free
OPENROUTER_API_KEY=replace_me_openrouter_key
```

For fast text only:

```text
LLM_PROVIDER=groq
GROQ_TEXT_MODEL=llama-3.1-8b-instant
GROQ_API_KEY=replace_me_groq_key
```

## Paid Production Recommendation

For production-grade JSON adherence and better vision judgement, use Claude or OpenAI. Keep the deterministic OpenCV vessel-damage candidates as the source of coordinates, then let the vision LLM explain, rank, or reject them.

```text
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=replace_me_anthropic_key
ANTHROPIC_TEXT_MODEL=claude-sonnet-4-5
ANTHROPIC_VISION_MODEL=claude-sonnet-4-5
```

## Where To Put Keys

Root backend keys:

```text
retinascope-ai-v2/.env
```

Frontend public backend URL:

```text
retinascope-ai-v2/frontend/.env.local
```

Never put real keys in source files.

## Sources Checked

- OpenRouter free model router: https://openrouter.ai/openrouter/free/providers
- OpenRouter free models collection: https://openrouter.ai/collections/free-models
- OpenRouter `:free` model variant docs: https://openrouter.ai/docs/routing/model-variants
- Groq free plan rate limits: https://console.groq.com/docs/rate-limits
- Ollama Llama 3.2 Vision registry page: https://registry.ollama.ai/library/llama3.2-vision
- Hugging Face model pages: https://hf.co/Qwen/Qwen2.5-VL-7B-Instruct and https://hf.co/meta-llama/Llama-3.2-11B-Vision-Instruct
