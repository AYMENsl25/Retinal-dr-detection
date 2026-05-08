// MOCK BACKEND — replace with calls to your FastAPI service.
// Contract is the AnalyzeResponse type in lib/types.ts.
// Phase 3 of PROJECT_PLAN.md: POST /api/v1/analyze

import { NextResponse } from "next/server"
import type { AnalyzeResponse, DRGrade } from "@/lib/types"

export const runtime = "nodejs"

function randIn(min: number, max: number) {
  return Math.random() * (max - min) + min
}

function softmaxFromGrade(grade: DRGrade): [number, number, number, number, number] {
  // Build a peaked distribution around `grade` with a bit of mass on neighbors.
  const raw = [0, 1, 2, 3, 4].map((g) => {
    const d = Math.abs(g - grade)
    return Math.exp(-d * 1.6) + Math.random() * 0.05
  })
  const sum = raw.reduce((a, b) => a + b, 0)
  return raw.map((v) => v / sum) as [number, number, number, number, number]
}

const CLINICAL_BY_GRADE: Record<DRGrade, { summary: string; pathophys: string; followUp: string }> = {
  0: {
    summary:
      "No diabetic retinopathy detected. Retinal vasculature appears within normal limits with no microaneurysms, hemorrhages, or exudates identified.",
    pathophys:
      "The retinal microvasculature shows no signs of diabetes-induced damage. The blood-retinal barrier appears intact and capillary perfusion is preserved.",
    followUp: "Annual dilated fundus examination",
  },
  1: {
    summary:
      "Mild non-proliferative diabetic retinopathy (NPDR). A small number of microaneurysms are present without other lesions. Vessel architecture is largely preserved.",
    pathophys:
      "Hyperglycemia-induced loss of pericytes leads to focal capillary outpouchings (microaneurysms). At this stage the blood-retinal barrier remains mostly competent.",
    followUp: "Repeat dilated examination in 9–12 months",
  },
  2: {
    summary:
      "Moderate NPDR. Multiple microaneurysms, dot-blot hemorrhages, and possible cotton-wool spots are observed. Vessel tortuosity is mildly increased.",
    pathophys:
      "Progressive capillary closure leads to focal ischemia. Cotton-wool spots represent nerve-fiber-layer infarcts; venous beading begins to appear.",
    followUp: "Repeat dilated examination in 6 months",
  },
  3: {
    summary:
      "Severe NPDR. Extensive intraretinal hemorrhages in all four quadrants and/or definite venous beading and intraretinal microvascular abnormalities (IRMA).",
    pathophys:
      "Widespread capillary non-perfusion drives VEGF upregulation. Risk of progression to proliferative disease within one year is approximately 50%.",
    followUp: "Refer to retina specialist within 1 month",
  },
  4: {
    summary:
      "Proliferative diabetic retinopathy. Neovascularization is suspected at the disc and/or elsewhere, with high risk of vitreous hemorrhage and tractional retinal detachment.",
    pathophys:
      "VEGF-driven neovascularization on a fragile fibrovascular scaffold. Sight-threatening complications (vitreous hemorrhage, traction) are imminent without intervention.",
    followUp: "Urgent referral — retina specialist within 1–2 weeks",
  },
}

export async function POST(req: Request) {
  // Accept multipart/form-data with field "image" — we don't actually run a model here,
  // we just simulate latency and return a plausible JSON payload.
  const form = await req.formData().catch(() => null)
  const file = form?.get("image")
  if (!file || !(file instanceof File)) {
    return NextResponse.json({ error: "Missing image file" }, { status: 400 })
  }

  // Simulate model + LLM latency.
  await new Promise((r) => setTimeout(r, 1200))

  // Pick a grade biased toward 1–2 (typical screening cohort).
  const roll = Math.random()
  const grade: DRGrade = roll < 0.15 ? 0 : roll < 0.55 ? 1 : roll < 0.8 ? 2 : roll < 0.95 ? 3 : 4

  const probs = softmaxFromGrade(grade)
  const calibrated = probs[grade]
  const nextIdx = Math.min(grade + 1, 4)
  const closeness = probs[nextIdx] / (probs[grade] + probs[nextIdx])

  const entropy = -probs.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0)
  const mc_std = randIn(0.04, 0.14)

  let decision_flag: AnalyzeResponse["decision_flag"] = "HIGH_CONFIDENCE"
  if (entropy >= 0.6 || grade >= 3) decision_flag = "REFER_SPECIALIST"
  else if (entropy >= 0.3 || closeness > 0.4) decision_flag = "MEDIUM_REFER_RECOMMENDED"

  const clin = CLINICAL_BY_GRADE[grade]
  const damage_score = Math.round(grade * 22 + randIn(-6, 6) + 8)

  const payload: AnalyzeResponse = {
    case_id: crypto.randomUUID(),
    panels: { original: "" }, // computed client-side from upload
    grade,
    grade_probs: probs,
    calibrated_confidence: calibrated,
    closeness_to_next_grade: closeness,
    uncertainty: { entropy, mc_dropout_std: mc_std },
    biomarkers: {
      vessel_density: +randIn(0.12, 0.22).toFixed(3),
      tortuosity: +randIn(1.1, 1.6).toFixed(3),
      fractal_dim: +randIn(1.55, 1.72).toFixed(3),
      avr: +randIn(0.6, 0.78).toFixed(3),
    },
    decision_flag,
    clinical_report: {
      summary: clin.summary,
      pathophysiology: clin.pathophys,
      risk_factors: [
        "Duration of diabetes mellitus",
        "Glycemic control (HbA1c)",
        "Coexisting hypertension",
        "Dyslipidemia",
        "Pregnancy (if applicable)",
      ],
      recommendations: [
        "Optimize glycemic control — target HbA1c < 7.0% per ADA guidance",
        "Strict blood-pressure control (target < 130/80 mmHg)",
        "Lipid management with statin therapy if indicated",
        grade >= 2 ? "Consider OCT and fluorescein angiography" : "Routine retinal photography sufficient",
      ],
      follow_up_window: clin.followUp,
      lifestyle_advice: [
        "Smoking cessation",
        "Regular aerobic exercise (≥150 min/week)",
        "Mediterranean-style diet rich in leafy greens",
      ],
      red_flags: [
        "Sudden vision loss",
        "New floaters or flashes of light",
        "Distortion of central vision",
      ],
      disclaimer:
        "Decision support only — not a medical diagnosis. All findings must be confirmed by a licensed ophthalmologist.",
    },
    vascular_report: {
      damaged_regions:
        grade === 0
          ? []
          : [
              {
                quadrant: "superior-temporal",
                severity: grade >= 3 ? "severe" : grade >= 2 ? "moderate" : "mild",
                finding: "Microaneurysms clustered along secondary arterioles",
              },
              {
                quadrant: "inferior-nasal",
                severity: grade >= 3 ? "severe" : "mild",
                finding: "Localized vessel attenuation and mild tortuosity",
              },
              ...(grade >= 2
                ? ([
                    {
                      quadrant: "macula",
                      severity: "moderate",
                      finding: "Dot-blot hemorrhages within 2 disc-diameters of fovea",
                    },
                  ] as const)
                : []),
              ...(grade >= 3
                ? ([
                    {
                      quadrant: "superior-nasal",
                      severity: "severe",
                      finding: "Venous beading and intraretinal microvascular abnormalities",
                    },
                  ] as const)
                : []),
            ],
      overall_damage_score: Math.max(0, Math.min(100, damage_score)),
      per_grade_score: {
        "0": Math.round(probs[0] * 100),
        "1": Math.round(probs[1] * 100),
        "2": Math.round(probs[2] * 100),
        "3": Math.round(probs[3] * 100),
        "4": Math.round(probs[4] * 100),
      },
      closeness_to_next_grade: closeness,
      needs_specialist_review: decision_flag !== "HIGH_CONFIDENCE",
      rationale:
        grade === 0
          ? "Vessel caliber, tortuosity, and fractal dimension fall within healthy reference ranges. No focal vascular damage identified."
          : `Quantitative biomarkers and segmentation patterns are consistent with ${CLINICAL_BY_GRADE[grade].summary.split(".")[0].toLowerCase()}. The vision-conditioned analysis cross-validated the CNN-assigned grade with the computed vessel-density and tortuosity metrics.`,
    },
  }

  return NextResponse.json(payload)
}
