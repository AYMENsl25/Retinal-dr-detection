import { generateText, Output } from "ai"
import { z } from "zod"
import { fallbackVascularReport } from "@/lib/fallback-reports"

export const maxDuration = 60

const VascularReportSchema = z.object({
  damaged_regions: z
    .array(
      z.object({
        x_min: z.number().int().min(0).max(512),
        y_min: z.number().int().min(0).max(512),
        x_max: z.number().int().min(0).max(512),
        y_max: z.number().int().min(0).max(512),
        quadrant: z.enum(["NW", "NE", "SW", "SE"]),
        severity: z.enum(["low", "medium", "high"]),
        finding: z.string().max(60),
      }),
    )
    .max(8),
  overall_damage_score: z.number().int().min(0).max(100),
  per_grade_severity: z.object({
    "0": z.number().min(0).max(1),
    "1": z.number().min(0).max(1),
    "2": z.number().min(0).max(1),
    "3": z.number().min(0).max(1),
    "4": z.number().min(0).max(1),
  }),
  needs_specialist_review: z.boolean(),
  rationale: z.string(),
  closeness_to_next_grade: z.number().min(0).max(1),
})

export async function POST(req: Request) {
  const { grade, biomarkers, candidate_regions } = await req.json()
  try {

    const candidateBlock =
      candidate_regions && candidate_regions.length > 0
        ? `\nCandidate damaged regions detected by skeleton/connectivity analysis (you may keep, refine, or discard these):\n${JSON.stringify(
            candidate_regions,
            null,
            2,
          )}`
        : "\nNo strong candidate regions detected by skeleton/connectivity analysis."

    const prompt = `You are a retinal vascular damage analyst. You are reviewing a 512×512 binary vessel segmentation mask (white vessels on black background) from a diabetic retinopathy screening pipeline. You will not see the image directly — instead you have quantitative biomarkers and pre-computed candidate damaged regions extracted by OpenCV skeletonization and connectivity analysis.

Your task: produce a structured vascular damage report. Identify regions where vessel structure appears DISCONTINUOUS, ABNORMALLY TORTUOUS, CALIBER IRREGULAR, or FOCALLY ABSENT. Stay grounded in the biomarkers — do NOT invent regions that contradict the data.

Inputs:
- ICDR DR grade (from grader CNN): ${grade}
- Vessel density: ${biomarkers.vessel_density}
- Tortuosity index: ${biomarkers.tortuosity} (>1.5 is considered abnormal)
- Fractal dimension: ${biomarkers.fractal_dim}
- AVR: ${biomarkers.avr}
- Num vessel components: ${biomarkers.num_vessel_components}
- Estimated broken segments: ${biomarkers.num_broken_segments_estimate}
- Quadrant density: NW=${biomarkers.quadrant_density.NW}, NE=${biomarkers.quadrant_density.NE}, SW=${biomarkers.quadrant_density.SW}, SE=${biomarkers.quadrant_density.SE}
${candidateBlock}

Rules:
- All bounding-box coordinates are pixel integers in the 512×512 mask. Clamp to [0, 512].
- Return at most 8 damaged_regions. If no damage is visible, return an empty array.
- overall_damage_score ∈ [0, 100]. Roughly: 0–15 = none/normal, 15–40 = mild, 40–70 = moderate, 70–100 = severe.
- per_grade_severity values must sum to ~1.0 — these reflect your visual confidence in each ICDR grade.
- needs_specialist_review = true if overall_damage_score > 60 OR ICDR grade ≥ 3 OR your visual analysis disagrees with the CNN grade.
- rationale: 1–2 sentences grounded in the biomarkers.`

    const { experimental_output } = await generateText({
      model: "anthropic/claude-sonnet-4.6",
      prompt,
      experimental_output: Output.object({ schema: VascularReportSchema }),
      maxOutputTokens: 1200,
      temperature: 0.15,
    })

    return Response.json(experimental_output)
  } catch (err) {
    console.warn(
      "[v0] vascular-report: LLM unavailable, using deterministic fallback.",
      err instanceof Error ? err.message : err,
    )
    const fallback = fallbackVascularReport({
      grade,
      biomarkers,
      candidate_regions: candidate_regions ?? [],
    })
    return Response.json(fallback)
  }
}
