"use client"

import { useState } from "react"
import { Loader2, PlayCircle, RotateCcw } from "lucide-react"
import { toast } from "sonner"
import { SiteHeader } from "@/components/site-header"
import { UploadDropzone } from "@/components/upload-dropzone"
import { FourPanelViewer } from "@/components/four-panel-viewer"
import { GradeCard } from "@/components/grade-card"
import { ClinicalReport } from "@/components/clinical-report"
import { VascularReport } from "@/components/vascular-report"
import { ConsultationChat } from "@/components/consultation-chat"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { generatePanels, type PanelImages } from "@/lib/image-processing"
import type { AnalyzeResponse } from "@/lib/types"

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null)
  const [panels, setPanels] = useState<PanelImages | null>(null)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [running, setRunning] = useState(false)

  async function handleAnalyze() {
    if (!file) return
    setRunning(true)
    setPanels(null)
    setResult(null)

    try {
      // Run client-side panel generation and the (mock) backend in parallel.
      const [panelImages, response] = await Promise.all([
        generatePanels(file),
        (async () => {
          const fd = new FormData()
          fd.append("image", file)
          const res = await fetch("/api/analyze", { method: "POST", body: fd })
          if (!res.ok) throw new Error("Analysis failed")
          return (await res.json()) as AnalyzeResponse
        })(),
      ])
      setPanels(panelImages)
      setResult(response)
      toast.success("Analysis complete", {
        description: `Grade ${response.grade} · ${(response.calibrated_confidence * 100).toFixed(0)}% confidence`,
      })
    } catch (err) {
      console.log("[v0] analyze failed", err)
      toast.error("Analysis failed", {
        description: "Check the console — the mock backend or canvas processing errored.",
      })
    } finally {
      setRunning(false)
    }
  }

  function handleReset() {
    setFile(null)
    setPanels(null)
    setResult(null)
  }

  return (
    <div className="min-h-screen bg-background">
      <SiteHeader patientId={result?.case_id} />

      <main className="mx-auto max-w-7xl px-4 md:px-6 py-6 space-y-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-balance">Fundus Analysis</h1>
            <p className="text-sm text-muted-foreground text-pretty">
              Upload a single retinal fundus image. The pipeline returns vessel segmentation, an
              ICDR grade, calibrated confidence, and dual-LLM clinical reasoning.
            </p>
          </div>
          {result ? (
            <Button variant="outline" size="sm" onClick={handleReset}>
              <RotateCcw className="h-3.5 w-3.5" />
              New analysis
            </Button>
          ) : null}
        </div>

        {!result ? (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">1 · Upload fundus image</CardTitle>
              <CardDescription>
                Color fundus photography, single eye. The mock pipeline simulates the stages
                described in §5 of the project plan.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <UploadDropzone file={file} onFileChange={setFile} disabled={running} />
              <div className="flex items-center justify-between gap-3">
                <PipelineSteps running={running} />
                <Button onClick={handleAnalyze} disabled={!file || running}>
                  {running ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Running pipeline…
                    </>
                  ) : (
                    <>
                      <PlayCircle className="h-4 w-4" />
                      Run analysis
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>
        ) : null}

        {result && panels ? (
          <>
            <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
              <FourPanelViewer panels={panels} />
              <GradeCard result={result} />
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <ClinicalReport report={result.clinical_report} />
              <VascularReport report={result.vascular_report} />
            </div>

            <ConsultationChat caseContext={result} />
          </>
        ) : null}
      </main>
    </div>
  )
}

function PipelineSteps({ running }: { running: boolean }) {
  const steps = ["Preprocess", "Segment", "Grade", "Reason"]
  return (
    <ol className="hidden md:flex items-center gap-2 text-[11px] text-muted-foreground">
      {steps.map((s, i) => (
        <li key={s} className="flex items-center gap-2">
          <span
            className={`flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-mono ${
              running ? "bg-primary/10 text-primary animate-pulse" : "bg-muted"
            }`}
          >
            {i + 1}
          </span>
          <span>{s}</span>
          {i < steps.length - 1 ? <span aria-hidden="true">→</span> : null}
        </li>
      ))}
    </ol>
  )
}
