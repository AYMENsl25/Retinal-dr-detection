import { streamText, convertToModelMessages, type UIMessage } from "ai"

export const maxDuration = 60

interface CaseContext {
  case_id?: string
  grade?: number
  grade_name?: string
  confidence?: number
  closeness?: number
  decision_flag?: string
  biomarkers?: Record<string, unknown>
  clinical_summary?: string
  vascular_rationale?: string
  damage_score?: number
}

export async function POST(req: Request) {
  try {
    const {
      messages,
      context,
    }: { messages: UIMessage[]; context?: CaseContext } = await req.json()

    const ctxBlock = context
      ? `Current case context:
- Case ID: ${context.case_id ?? "—"}
- ICDR DR Grade: ${context.grade ?? "—"} (${context.grade_name ?? "—"})
- Calibrated confidence: ${context.confidence != null ? Math.round(context.confidence * 100) + "%" : "—"}
- Closeness to next grade: ${context.closeness != null ? Math.round(context.closeness * 100) + "%" : "—"}
- Decision flag: ${context.decision_flag ?? "—"}
- Vascular damage score: ${context.damage_score ?? "—"}/100
- Clinical summary: ${context.clinical_summary ?? "—"}
- Vascular rationale: ${context.vascular_rationale ?? "—"}
- Biomarkers: ${JSON.stringify(context.biomarkers ?? {})}`
      : "No case context yet."

    const system = `You are RetinaScope-AI's clinical consultation assistant — a specialized assistant for diabetic retinopathy decision support. You answer follow-up questions about the current retinal-imaging case ONLY, using the structured context provided. You speak with the precision and brevity of an ophthalmologist consultant.

Rules:
- Stay strictly scoped to this case. Politely decline unrelated medical questions.
- Be medically accurate, grounded in the ICDR scale and AAO Preferred Practice Pattern.
- Always close with a brief reminder this is decision support, not a diagnosis.
- Keep replies under 140 words unless the user explicitly asks for more detail.
- Never invent biomarker values not present in the context.

${ctxBlock}`

    const result = streamText({
      model: "anthropic/claude-sonnet-4.6",
      system,
      messages: await convertToModelMessages(messages),
      maxOutputTokens: 800,
      temperature: 0.3,
    })

    return result.toUIMessageStreamResponse({
      onError: (error) => {
        console.warn("[v0] chat stream error:", error)
        const msg =
          error instanceof Error ? error.message : "Unknown chat error"
        if (
          msg.includes("credit card") ||
          msg.includes("customer_verification_required") ||
          msg.includes("403")
        ) {
          return "The consultation assistant is temporarily offline because the AI Gateway is not configured (add a credit card or AI_GATEWAY_API_KEY). The deterministic clinical and vascular reports above remain valid."
        }
        return `Consultation assistant error: ${msg}`
      },
    })
  } catch (err) {
    console.error("[v0] chat error:", err)
    return Response.json({ error: "Chat failed" }, { status: 500 })
  }
}
