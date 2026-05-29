"use client"

import { useCallback, useState } from "react"
import { RotateCcw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { toast } from "sonner"

import { SiteHeader } from "@/components/site-header"
import { UploadDropzone } from "./upload-dropzone"
import { PipelineLoader } from "./pipeline-loader"
import { PanelViewer } from "./panel-viewer"
import { DamageZoomStrip } from "./damage-zoom-strip"
import { GradeCard } from "./grade-card"
import { ClinicalReport } from "./clinical-report"
import { VascularReport } from "./vascular-report"
import { ConsultationChat } from "./consultation-chat"
import { DecisionBanner } from "./decision-banner"

import {
  buildMockDamageRegions,
  type PipelineStep,
} from "@/lib/mock-pipeline"
import { runAnalysisPipeline } from "@/lib/api"
import type {
  AnalysisResult,
  ClinicalReportData,
  VascularReportData,
} from "@/lib/types"

type Phase = "upload" | "running" | "results"

export function AnalyzeFlow() {
  const [phase, setPhase] = useState<Phase>("upload")
  const [step, setStep] = useState<PipelineStep | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)

  const [clinical, setClinical] = useState<ClinicalReportData | null>(null)
  const [clinicalLoading, setClinicalLoading] = useState(false)
  const [clinicalError, setClinicalError] = useState<string | null>(null)

  const [vascular, setVascular] = useState<VascularReportData | null>(null)
  const [vascularLoading, setVascularLoading] = useState(false)
  const [vascularError, setVascularError] = useState<string | null>(null)

  const reset = useCallback(() => {
    setPhase("upload")
    setStep(null)
    setResult(null)
    setClinical(null)
    setClinicalError(null)
    setVascular(null)
    setVascularError(null)
  }, [])

  const startAnalysis = useCallback(async (dataUrl: string, file: File) => {
    setPhase("running")
    setStep("preprocess")
    setClinical(null)
    setVascular(null)
    setClinicalError(null)
    setVascularError(null)

    try {
      const analysis = await runAnalysisPipeline({
        imageDataUrl: dataUrl,
        file,
        onStep: (s) => setStep(s),
      })
      setResult(analysis)
      setPhase("results")
      toast.success(`Analysis complete — Case ${analysis.case_id}`, {
        description: `File ${file.name} processed in 4 stages.`,
      })

      if (analysis.clinical_report) {
        setClinical(analysis.clinical_report)
      }
      if (analysis.vascular_report) {
        setVascular(analysis.vascular_report)
      }

      // Kick off local Next.js LLM routes only when the backend did not return reports.
      setClinicalLoading(!analysis.clinical_report)
      setVascularLoading(!analysis.vascular_report)

      const candidateRegions = buildMockDamageRegions(analysis.panels.crops)

      if (!analysis.clinical_report) fetch("/api/clinical-report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          grade: analysis.grade,
          biomarkers: analysis.biomarkers,
          confidence: analysis.calibrated_confidence,
          closeness: analysis.closeness_to_next_grade,
        }),
      })
        .then(async (r) => {
          if (!r.ok) throw new Error(`Clinical report failed (${r.status})`)
          return r.json()
        })
        .then((data: ClinicalReportData) => setClinical(data))
        .catch((e: Error) => {
          console.error("[v2] clinical fetch error", e)
          setClinicalError(
            "Could not generate the clinical narrative. The model may be temporarily unavailable.",
          )
        })
        .finally(() => setClinicalLoading(false))

      if (!analysis.vascular_report) fetch("/api/vascular-report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          grade: analysis.grade,
          biomarkers: analysis.biomarkers,
          candidate_regions: candidateRegions,
        }),
      })
        .then(async (r) => {
          if (!r.ok) throw new Error(`Vascular report failed (${r.status})`)
          return r.json()
        })
        .then((data: VascularReportData) => setVascular(data))
        .catch((e: Error) => {
          console.error("[v2] vascular fetch error", e)
          setVascularError(
            "Could not generate the vascular damage report. The model may be temporarily unavailable.",
          )
        })
        .finally(() => setVascularLoading(false))
    } catch (err) {
      console.error("[v2] pipeline error", err)
      toast.error("Pipeline failed", {
        description:
          err instanceof Error ? err.message : "Unknown error during analysis.",
      })
      setPhase("upload")
    }
  }, [])

  return (
    <>
      <SiteHeader
        caseId={result?.case_id ?? null}
        showExport={phase === "results"}
        onExport={() => window.print()}
      />
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {phase === "upload" && (
          <div className="mx-auto max-w-2xl">
            <div className="mb-6">
              <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
                Fundus Analysis
              </h1>
              <p className="mt-1.5 text-sm text-muted-foreground">
                Upload a single retinal fundus image to receive vessel
                segmentation, ICDR grading, calibrated confidence, vascular
                damage localization, and dual-LLM clinical reasoning.
              </p>
            </div>
            <UploadDropzone onSelect={startAnalysis} />
          </div>
        )}

        {phase === "running" && <PipelineLoader step={step} />}

        {phase === "results" && result && (
          <div className="space-y-5">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div>
                <h1 className="text-2xl font-bold tracking-tight">
                  Fundus Analysis
                </h1>
                <p className="text-sm text-muted-foreground">
                  Case{" "}
                  <span className="font-mono font-semibold text-foreground">
                    {result.case_id}
                  </span>{" "}
                  · {new Date(result.created_at).toLocaleString()}
                </p>
              </div>
              <Button variant="outline" size="sm" onClick={reset}>
                <RotateCcw className="h-4 w-4" aria-hidden="true" />
                New analysis
              </Button>
            </div>

            <DecisionBanner
              flag={result.decision_flag}
              entropy={result.uncertainty.entropy}
            />

            <div className="grid gap-5 lg:grid-cols-[1fr_320px]">
              <div className="rounded-xl border border-border bg-card p-4 sm:p-5">
                <PanelViewer panels={result.panels} />
                <DamageZoomStrip crops={result.panels.crops} />
              </div>
              <GradeCard result={result} />
            </div>

            <div className="grid gap-5 lg:grid-cols-2">
              <ClinicalReport
                loading={clinicalLoading}
                report={clinical}
                error={clinicalError}
              />
              <VascularReport
                loading={vascularLoading}
                report={vascular}
                error={vascularError}
              />
            </div>

            <ConsultationChat
              result={result}
              clinical={clinical}
              vascular={vascular}
            />

            <p className="border-t border-border pt-4 text-center text-[11px] text-muted-foreground">
              Decision support only — not a medical diagnosis. All findings must
              be confirmed by a licensed ophthalmologist.
            </p>
          </div>
        )}
      </main>
    </>
  )
}
