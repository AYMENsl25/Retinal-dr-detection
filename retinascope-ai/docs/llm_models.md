# LLM Model Choices

As of the May 2026 build, the free-test stack is:

| Role | Provider | Model | Key |
| --- | --- | --- | --- |
| Clinical Explainer | Groq | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| Vascular Vision Analyst | Google Gemini API | `gemini-3.1-flash-lite` | `GEMINI_API_KEY` |
| Consultation Chat | Google Gemini API | `gemini-3.1-flash-lite` | `GEMINI_API_KEY` |
| Optional fallback | OpenRouter | `openrouter/free` | `OPENROUTER_API_KEY` |

Why:

- Groq's Llama 3.3 70B model supports JSON object mode and tool use, useful for structured clinical reports.
- Gemini 3.1 Flash-Lite is the high-throughput Gemini choice for chat and multimodal report generation.
- OpenRouter's `openrouter/free` router is useful as a demo fallback when a provider is rate-limited.

Privacy warning: free tiers are for synthetic or public test images only. Use paid zero-retention terms or local Ollama before processing real patient data.
