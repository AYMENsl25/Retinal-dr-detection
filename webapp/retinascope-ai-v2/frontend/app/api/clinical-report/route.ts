import { generateText, Output } from "ai"
import { z } from "zod"
import { fallbackClinicalReport } from "@/lib/fallback-reports"

export const maxDuration = 60

const ClinicalReportSchema = z.object({
  summary: z.string().describe("1-2 sentence high-level summary of the case."),
  pathophysiology: z
    .string()
    .describe("2-3 sentence medical explanation grounded in ICDR and AAO PPP."),
  risk_factors: z.array(z.string()).min(2).max(6),
  recommendations: z.array(z.string()).min(2).max(6),
  follow_up_window: z
    .string()
    .describe("Time window for next screening, e.g. '12 months' or '1 month'."),
  lifestyle_advice: z.array(z.string()).min(2).max(6),
  red_flags: z.array(z.string()).min(0).max(5),
  disclaimer: z
    .string()
    .describe(
      "Standard decision-support disclaimer reminding this is not a diagnosis.",
    ),
})

const GRADE_NAMES = [
  "No DR",
  "Mild NPDR",
  "Moderate NPDR",
  "Severe NPDR",
  "Proliferative DR",
]

export async function POST(req: Request) {
  const { grade, biomarkers, confidence, closeness } = await req.json()
  const gradeName = GRADE_NAMES[grade] ?? "Unknown"

  try {

    const prompt = `You are a senior ophthalmology clinical-decision-support assistant generating a structured report for a diabetic retinopathy screening result. Ground your output strictly in the International Clinical Diabetic Retinopathy (ICDR) severity scale and the American Academy of Ophthalmology Preferred Practice Pattern (AAO PPP).

Case data:
- ICDR Grade: ${grade} — ${gradeName}
- Calibrated confidence: ${(confidence * 100).toFixed(1)}%
- Closeness to next grade: ${(closeness * 100).toFixed(1)}%
- Vessel density: ${biomarkers.vessel_density}
- Tortuosity index: ${biomarkers.tortuosity}
- Fractal dimension: ${biomarkers.fractal_dim}
- AVR (arteriolar-venular ratio): ${biomarkers.avr}
- Estimated broken vessel segments: ${biomarkers.num_broken_segments_estimate}

Produce a clinician-facing report. Be specific, concise, and medically accurate. The follow_up_window must reflect ICDR guidance (e.g. No DR → 12 months; Mild NPDR → 6–12 months; Moderate → 3–6 months; Severe/PDR → urgent referral within 1–4 weeks). Always include a disclaimer that this output is decision support only and must be confirmed by a licensed ophthalmologist.`

    const { experimental_output } = await generateText({
      model: "anthropic/claude-sonnet-4.6",
      prompt,
      experimental_output: Output.object({ schema: ClinicalReportSchema }),
      maxOutputTokens: 1500,
      temperature: 0.2,
    })

    return Response.json(experimental_output)
  } catch (err) {
    console.warn(
      "[v0] clinical-report: LLM unavailable, using deterministic fallback.",
      err instanceof Error ? err.message : err,
    )
    const fallback = fallbackClinicalReport({
      grade,
      biomarkers,
      confidence,
      closeness,
    })
    return Response.json(fallback)
  }
}
