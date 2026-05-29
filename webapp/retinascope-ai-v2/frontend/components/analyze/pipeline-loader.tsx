"use client"

import { Check } from "lucide-react"
import { Spinner } from "@/components/ui/spinner"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import type { PipelineStep } from "@/lib/mock-pipeline"

const STEPS: { id: PipelineStep; label: string; detail: string }[] = [
  {
    id: "preprocess",
    label: "Preprocessing",
    detail: "CLAHE · ROI crop · resize 512×512 · normalize",
  },
  {
    id: "segment",
    label: "Vessel segmentation",
    detail: "U-Net ensemble → binary vessel mask",
  },
  {
    id: "grade",
    label: "DR grading",
    detail: "Grader CNN → temperature-scaled softmax",
  },
  {
    id: "reason",
    label: "Vascular & clinical reasoning",
    detail: "Skeleton analysis · LLM-1 + LLM-2 in parallel",
  },
]

interface PipelineLoaderProps {
  step: PipelineStep | null
}

export function PipelineLoader({ step }: PipelineLoaderProps) {
  const idx = step ? STEPS.findIndex((s) => s.id === step) : 0

  return (
    <div className="mx-auto max-w-xl py-12">
      <div className="flex flex-col items-center text-center">
        <Spinner className="h-10 w-10 text-primary" />
        <h2 className="mt-6 text-xl font-semibold">Running analysis pipeline…</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          This typically takes a few seconds. Do not close the window.
        </p>
      </div>

      <Card className="mt-8 border-border">
        <CardContent className="p-5">
          <ol className="space-y-3">
            {STEPS.map((s, i) => {
              const done = i < idx
              const active = i === idx
              return (
                <li key={s.id} className="flex items-start gap-3">
                  <div
                    className={cn(
                      "mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border text-[10px] font-bold",
                      done && "border-primary bg-primary text-primary-foreground",
                      active && "border-primary bg-accent text-primary",
                      !done && !active && "border-border text-muted-foreground",
                    )}
                  >
                    {done ? (
                      <Check className="h-3.5 w-3.5" aria-hidden="true" />
                    ) : active ? (
                      <span className="block h-2 w-2 animate-pulse rounded-full bg-primary" />
                    ) : (
                      i + 1
                    )}
                  </div>
                  <div className="flex-1">
                    <div
                      className={cn(
                        "text-sm font-medium",
                        done || active ? "text-foreground" : "text-muted-foreground",
                      )}
                    >
                      {s.label}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {s.detail}
                    </div>
                  </div>
                  {done && (
                    <span className="text-xs font-mono text-primary">done</span>
                  )}
                  {active && (
                    <span className="text-xs font-mono text-primary">
                      running
                    </span>
                  )}
                </li>
              )
            })}
          </ol>
        </CardContent>
      </Card>
    </div>
  )
}
