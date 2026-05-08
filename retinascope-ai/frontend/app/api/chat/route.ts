// MOCK STREAMING CHAT — replace with your LLM provider call.
// Phase 8 of PROJECT_PLAN.md: POST /api/v1/chat
// Streams a plausible clinician-style answer scoped to the case context.

import type { NextRequest } from "next/server"
import type { ChatMessage, AnalyzeResponse } from "@/lib/types"

export const runtime = "nodejs"

interface ChatBody {
  messages: ChatMessage[]
  caseContext: AnalyzeResponse | null
}

function buildAnswer(question: string, ctx: AnalyzeResponse | null): string {
  const q = question.toLowerCase()
  const grade = ctx?.grade ?? 1
  const labelByGrade = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"][grade]

  if (q.includes("follow") || q.includes("when") || q.includes("next")) {
    return `Based on the current finding of **${labelByGrade}**, the recommended follow-up window is **${ctx?.clinical_report.follow_up_window ?? "9–12 months"}**. If the patient develops new visual symptoms (sudden loss, floaters, flashes), they should be re-evaluated immediately rather than waiting for the scheduled visit.`
  }
  if (q.includes("treat") || q.includes("medication") || q.includes("therap")) {
    return `At ${labelByGrade}, the cornerstone of treatment is systemic optimization: tight glycemic control (HbA1c target < 7.0%), blood-pressure management (< 130/80 mmHg), and lipid control. Intravitreal anti-VEGF therapy or focal/grid laser is reserved for diabetic macular edema or proliferative disease — both should be confirmed with OCT before initiating.`
  }
  if (q.includes("confidence") || q.includes("uncertain") || q.includes("how sure")) {
    return `The calibrated softmax confidence is **${((ctx?.calibrated_confidence ?? 0.7) * 100).toFixed(0)}%** with predictive entropy of ${(ctx?.uncertainty.entropy ?? 0.4).toFixed(2)}. The closeness-to-next-grade score of ${((ctx?.closeness_to_next_grade ?? 0) * 100).toFixed(0)}% indicates how borderline this case is. Anything above ~30% closeness warrants a second look by a specialist.`
  }
  if (q.includes("biomarker") || q.includes("vessel") || q.includes("density") || q.includes("tortuos")) {
    const b = ctx?.biomarkers
    return `The quantitative vascular biomarkers for this case are: vessel density ${b?.vessel_density ?? "—"}, mean tortuosity ${b?.tortuosity ?? "—"}, fractal dimension ${b?.fractal_dim ?? "—"}, and arteriolar-to-venular ratio (AVR) ${b?.avr ?? "—"}. Reduced vessel density and elevated tortuosity are classic markers of progressive diabetic microvascular damage.`
  }
  if (q.includes("disclaim") || q.includes("legal") || q.includes("liab")) {
    return `This system is a **decision-support tool**, not a diagnostic device. All findings must be confirmed by a licensed ophthalmologist before any clinical action is taken. The model's output is non-binding and should be interpreted alongside the patient's full history, slit-lamp examination, and ancillary tests.`
  }

  return `For a patient classified as **${labelByGrade}**, the priority is to (1) confirm the grade with a dilated fundus examination, (2) evaluate for diabetic macular edema with OCT, and (3) reinforce systemic risk-factor control. Would you like me to elaborate on any specific aspect — treatment, follow-up cadence, or the underlying biomarkers?`
}

export async function POST(req: NextRequest) {
  const body = (await req.json()) as ChatBody

  const backendUrl = process.env.BACKEND_API_URL?.replace(/\/$/, "")
  if (backendUrl) {
    const upstream = await fetch(`${backendUrl}/api/v1/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("Content-Type") ?? "text/plain; charset=utf-8",
      },
    })
  }

  const last = body.messages.filter((m) => m.role === "user").pop()
  const answer = buildAnswer(last?.content ?? "", body.caseContext)

  // Token-by-token streaming simulation.
  const encoder = new TextEncoder()
  const stream = new ReadableStream({
    async start(controller) {
      const tokens = answer.match(/(\s+|[^\s]+)/g) ?? [answer]
      for (const t of tokens) {
        controller.enqueue(encoder.encode(t))
        await new Promise((r) => setTimeout(r, 18))
      }
      controller.close()
    },
  })

  return new Response(stream, {
    headers: { "Content-Type": "text/plain; charset=utf-8" },
  })
}
