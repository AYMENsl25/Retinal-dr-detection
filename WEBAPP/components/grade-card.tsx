import { AlertTriangle, CheckCircle2, ShieldAlert } from "lucide-react"
import type { AnalyzeResponse } from "@/lib/types"
import { GRADE_LABELS } from "@/lib/types"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"

const FLAG_META: Record<
  AnalyzeResponse["decision_flag"],
  { label: string; tone: string; Icon: typeof CheckCircle2 }
> = {
  HIGH_CONFIDENCE: {
    label: "High confidence",
    tone: "bg-primary/10 text-primary ring-primary/20",
    Icon: CheckCircle2,
  },
  MEDIUM_REFER_RECOMMENDED: {
    label: "Second look advised",
    tone: "bg-accent text-accent-foreground ring-accent-foreground/20",
    Icon: AlertTriangle,
  },
  REFER_SPECIALIST: {
    label: "Refer to specialist",
    tone: "bg-destructive/10 text-destructive ring-destructive/30",
    Icon: ShieldAlert,
  },
}

export function GradeCard({ result }: { result: AnalyzeResponse }) {
  const flag = FLAG_META[result.decision_flag]
  const FlagIcon = flag.Icon

  const nextGrade = Math.min(result.grade + 1, 4)
  const closenessPct = Math.round(result.closeness_to_next_grade * 100)
  const confidencePct = Math.round(result.calibrated_confidence * 100)

  return (
    <Card className="overflow-hidden">
      <CardHeader className="space-y-0 pb-3">
        <p className="text-[11px] uppercase tracking-wider text-muted-foreground">ICDR Grade</p>
        <CardTitle className="flex items-baseline gap-2">
          <span className="text-3xl font-bold tabular-nums">{result.grade}</span>
          <span className="text-base font-medium text-muted-foreground">
            {GRADE_LABELS[result.grade]}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div
          className={cn(
            "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1",
            flag.tone,
          )}
        >
          <FlagIcon className="h-3.5 w-3.5" aria-hidden="true" />
          <span>{flag.label}</span>
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Calibrated confidence</span>
            <span className="font-mono tabular-nums">{confidencePct}%</span>
          </div>
          <Progress value={confidencePct} aria-label="Calibrated confidence" />
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">
              Closeness to Grade {nextGrade}
            </span>
            <span className="font-mono tabular-nums">{closenessPct}%</span>
          </div>
          <Progress value={closenessPct} aria-label="Closeness to next grade" />
        </div>

        <dl className="grid grid-cols-2 gap-3 border-t border-border pt-3 text-xs">
          <div>
            <dt className="text-muted-foreground">Entropy</dt>
            <dd className="font-mono tabular-nums">{result.uncertainty.entropy.toFixed(2)}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">MC-Dropout σ</dt>
            <dd className="font-mono tabular-nums">{result.uncertainty.mc_dropout_std.toFixed(2)}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Vessel density</dt>
            <dd className="font-mono tabular-nums">{result.biomarkers.vessel_density}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Tortuosity</dt>
            <dd className="font-mono tabular-nums">{result.biomarkers.tortuosity}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Fractal dim.</dt>
            <dd className="font-mono tabular-nums">{result.biomarkers.fractal_dim}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">AVR</dt>
            <dd className="font-mono tabular-nums">{result.biomarkers.avr}</dd>
          </div>
        </dl>

        <div className="space-y-1 border-t border-border pt-3">
          <p className="text-[11px] uppercase tracking-wider text-muted-foreground">
            Per-grade probability
          </p>
          <div className="flex items-end gap-1 h-12">
            {result.grade_probs.map((p, i) => {
              const pct = Math.round(p * 100)
              const active = i === result.grade
              return (
                <div key={i} className="flex-1 flex flex-col items-center gap-1">
                  <div className="flex-1 w-full flex items-end">
                    <div
                      className={cn(
                        "w-full rounded-sm transition-all",
                        active ? "bg-primary" : "bg-muted-foreground/30",
                      )}
                      style={{ height: `${Math.max(2, pct)}%` }}
                      aria-label={`Grade ${i}: ${pct}%`}
                    />
                  </div>
                  <span
                    className={cn(
                      "text-[10px] font-mono tabular-nums",
                      active ? "font-semibold text-foreground" : "text-muted-foreground",
                    )}
                  >
                    {i}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
