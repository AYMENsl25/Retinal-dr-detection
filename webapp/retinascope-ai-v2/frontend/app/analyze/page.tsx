import type { Metadata } from "next"
import { AnalyzeFlow } from "@/components/analyze/analyze-flow"

export const metadata: Metadata = {
  title: "Analyze · RetinaScope-AI",
  description:
    "Upload a retinal fundus image to generate vessel segmentation, ICDR grading, calibrated confidence, vascular damage localization, and a structured clinical report.",
}

export default function AnalyzePage() {
  return <AnalyzeFlow />
}
