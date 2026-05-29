"use client"

import {
  Activity,
  CheckCircle2,
  AlertTriangle,
  Info,
  Waves,
} from "lucide-react"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"
import type { Severity, VascularReportData } from "@/lib/types"

interface VascularReportProps {
  loading: boolean
  report: VascularReportData | null
  error?: string | null
}

const SEVERITY_BADGE: Record<Severity, string> = {
  high: "bg-destructive/15 text-destructive border-destructive/30",
  medium: "bg-warning/15 text-warning-foreground border-warning/30",
  low: "bg-success/15 text-success border-success/30",
}

const GRADE_COLORS = [
  "[&>div]:bg-success",
  "[&>div]:bg-warning",
  "[&>div]:bg-warning",
  "[&>div]:bg-destructive",
  "[&>div]:bg-destructive",
]

export function VascularReport({ loading, report, error }: VascularReportProps) {
  const score = report?.overall_damage_score ?? 0
  const scoreBar =
    score > 60
      ? "[&>div]:bg-destructive"
      : score > 30
      ? "[&>div]:bg-warning"
      : "[&>div]:bg-primary"

  return (
    <Card className="border-border">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-md bg-accent text-accent-foreground">
              <Waves className="h-4 w-4" aria-hidden="true" />
            </span>
            <div>
              <div className="text-sm font-semibold">Vascular Damage Report</div>
              <div className="text-[11px] text-muted-foreground">
                LLM-2 · Vision-conditioned · biomarker-grounded
              </div>
            </div>
          </div>
          <Badge variant="secondary" className="text-[10px]">
            <Activity className="mr-1 h-3 w-3" aria-hidden="true" />
            Biomarker-grounded
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        {loading && (
          <div className="space-y-3" aria-busy="true" aria-live="polite">
            <Skeleton className="h-3 w-32" />
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-4/5" />
            <Skeleton className="h-3 w-2/3 mt-4" />
            <Skeleton className="h-3 w-full" />
          </div>
        )}

        {error && !loading && (
          <div className="rounded-md border border-destructive/30 bg-destructive/5 p-3 text-xs text-destructive">
            {error}
          </div>
        )}

        {report && !loading && (
          <>
            <div>
              <div className="mb-2 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                Overall damage score
              </div>
              <div className="flex items-center gap-3">
                <Progress
                  value={score}
                  className={cn("h-2 flex-1", scoreBar)}
                />
                <span className="font-mono text-2xl font-bold leading-none">
                  {score}
                  <span className="text-sm font-medium text-muted-foreground">
                    /100
                  </span>
                </span>
              </div>
            </div>

            <div>
              <div className="mb-2 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                Per-grade severity
              </div>
              <div className="space-y-1.5">
                {(["0", "1", "2", "3", "4"] as const).map((g) => {
                  const v = report.per_grade_severity[g] ?? 0
                  return (
                    <div key={g} className="flex items-center gap-2 text-xs">
                      <span className="w-12 font-mono text-muted-foreground">
                        Grade {g}
                      </span>
                      <div className="flex-1">
                        <Progress
                          value={v * 100}
                          className={cn("h-1.5", GRADE_COLORS[+g])}
                        />
                      </div>
                      <span className="w-10 text-right font-mono tabular-nums text-muted-foreground">
                        {(v * 100).toFixed(0)}%
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>

            {report.damaged_regions.length > 0 ? (
              <div>
                <div className="mb-2 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                  Damaged regions ({report.damaged_regions.length})
                </div>
                <ul className="space-y-1.5">
                  {report.damaged_regions.map((r, i) => (
                    <li
                      key={i}
                      className="flex items-center gap-2 rounded-md bg-secondary/40 px-2.5 py-1.5"
                    >
                      <span
                        className={cn(
                          "rounded border px-1.5 py-0.5 font-mono text-[9px] font-bold uppercase",
                          SEVERITY_BADGE[r.severity],
                        )}
                      >
                        {r.severity}
                      </span>
                      <span className="flex-1 text-xs text-foreground">
                        {r.finding}
                      </span>
                      <span className="font-mono text-[10px] text-muted-foreground">
                        {r.quadrant}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <div className="flex items-center gap-2 rounded-md bg-secondary/40 px-3 py-2 text-xs text-muted-foreground">
                <Info className="h-3.5 w-3.5" aria-hidden="true" />
                No focal vascular damage detected.
              </div>
            )}

            {report.rationale && (
              <div>
                <Separator />
                <div className="mt-4">
                  <div className="mb-1.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                    Rationale
                  </div>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    {report.rationale}
                  </p>
                </div>
              </div>
            )}

            {report.needs_specialist_review ? (
              <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/5 p-3 text-xs text-destructive">
                <AlertTriangle
                  className="mt-0.5 h-3.5 w-3.5 shrink-0"
                  aria-hidden="true"
                />
                <span>
                  Cross-check flagged disagreement between visual analysis and
                  CNN grade — specialist review recommended.
                </span>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-xs text-success">
                <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
                No specialist referral indicated at this time.
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}
