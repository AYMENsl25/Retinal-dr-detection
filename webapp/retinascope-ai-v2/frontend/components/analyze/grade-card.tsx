"use client"

import { AlertTriangle, AlertCircle, CheckCircle2 } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"
import { GRADE_NAMES, type AnalysisResult } from "@/lib/types"

interface GradeCardProps {
  result: AnalysisResult
}

const GRADE_TONE: Record<number, { text: string; bar: string; pill: string }> = {
  0: {
    text: "text-success",
    bar: "[&>div]:bg-success",
    pill: "bg-success/10 text-success border-success/30",
  },
  1: {
    text: "text-warning",
    bar: "[&>div]:bg-warning",
    pill: "bg-warning/10 text-warning-foreground border-warning/30",
  },
  2: {
    text: "text-warning",
    bar: "[&>div]:bg-warning",
    pill: "bg-warning/10 text-warning-foreground border-warning/30",
  },
  3: {
    text: "text-destructive",
    bar: "[&>div]:bg-destructive",
    pill: "bg-destructive/10 text-destructive border-destructive/30",
  },
  4: {
    text: "text-destructive",
    bar: "[&>div]:bg-destructive",
    pill: "bg-destructive/10 text-destructive border-destructive/30",
  },
}

export function GradeCard({ result }: GradeCardProps) {
  const {
    grade,
    grade_probs,
    calibrated_confidence,
    closeness_to_next_grade,
    uncertainty,
    biomarkers,
  } = result
  const tone = GRADE_TONE[grade]
  const gradeName = GRADE_NAMES[grade]

  return (
    <Card className="border-border">
      <CardContent className="space-y-5 p-5">
        <div>
          <div className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
            ICDR Grade
          </div>
          <div className="mt-1 flex items-baseline gap-3">
            <span
              className={cn("font-mono text-5xl font-bold leading-none", tone.text)}
            >
              {grade}
            </span>
            <span className="text-sm font-semibold text-foreground">
              {gradeName}
            </span>
          </div>

          <div className="mt-3">
            {grade >= 3 ? (
              <div className="inline-flex items-center gap-1.5 rounded-md border border-destructive/30 bg-destructive/10 px-2 py-1 text-xs font-semibold text-destructive">
                <AlertTriangle className="h-3.5 w-3.5" aria-hidden="true" />
                Refer to specialist
              </div>
            ) : grade >= 1 ? (
              <div className="inline-flex items-center gap-1.5 rounded-md border border-warning/30 bg-warning/10 px-2 py-1 text-xs font-semibold text-warning-foreground">
                <AlertCircle className="h-3.5 w-3.5" aria-hidden="true" />
                Second look advised
              </div>
            ) : (
              <div className="inline-flex items-center gap-1.5 rounded-md border border-success/30 bg-success/10 px-2 py-1 text-xs font-semibold text-success">
                <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
                No immediate referral
              </div>
            )}
          </div>
        </div>

        <div className="space-y-3">
          <div>
            <div className="mb-1.5 flex items-center justify-between text-xs">
              <span className="text-muted-foreground">Calibrated confidence</span>
              <span className="font-mono font-semibold tabular-nums">
                {Math.round(calibrated_confidence * 100)}%
              </span>
            </div>
            <Progress
              value={calibrated_confidence * 100}
              className={cn("h-1.5", tone.bar)}
            />
          </div>

          {grade < 4 && (
            <div>
              <div className="mb-1.5 flex items-center justify-between text-xs">
                <span className="text-muted-foreground">
                  Closeness to Grade {grade + 1}
                </span>
                <span className="font-mono font-semibold tabular-nums">
                  {Math.round(closeness_to_next_grade * 100)}%
                </span>
              </div>
              <Progress
                value={closeness_to_next_grade * 100}
                className="h-1.5 [&>div]:bg-muted-foreground"
              />
            </div>
          )}
        </div>

        <div className="border-t border-border pt-4">
          <div className="mb-2 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
            Per-grade probability
          </div>
          <div className="space-y-1.5">
            {grade_probs.map((p, i) => (
              <div key={i} className="flex items-center gap-2">
                <span
                  className={cn(
                    "w-3 font-mono text-[10px]",
                    i === grade ? "font-bold text-foreground" : "text-muted-foreground",
                  )}
                >
                  {i}
                </span>
                <div className="flex-1">
                  <Progress
                    value={p * 100}
                    className={cn(
                      "h-1",
                      i === grade ? tone.bar : "[&>div]:bg-muted-foreground/50",
                    )}
                  />
                </div>
                <span className="w-10 text-right font-mono text-[10px] tabular-nums text-muted-foreground">
                  {(p * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 border-t border-border pt-4 text-xs">
          <Metric label="Entropy" value={uncertainty.entropy.toFixed(2)} />
          <Metric label="MC-Dropout σ" value={uncertainty.mc_dropout_std.toFixed(3)} />
          <Metric
            label="Vessel density"
            value={biomarkers.vessel_density.toFixed(3)}
          />
          <Metric label="Tortuosity" value={biomarkers.tortuosity.toFixed(3)} />
          <Metric label="Fractal dim." value={biomarkers.fractal_dim.toFixed(3)} />
          <Metric label="AVR" value={biomarkers.avr.toFixed(3)} />
        </div>
      </CardContent>
    </Card>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-muted-foreground">{label}</div>
      <div className="mt-0.5 font-mono font-semibold tabular-nums text-foreground">
        {value}
      </div>
    </div>
  )
}
