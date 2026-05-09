import { Activity, Eye, MapPin, Sparkles } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"
import type { VascularReport as VascularReportT } from "@/lib/types"

const SEVERITY_TONE = {
  mild: "bg-primary/10 text-primary ring-primary/20",
  moderate: "bg-accent text-accent-foreground ring-accent-foreground/20",
  severe: "bg-destructive/10 text-destructive ring-destructive/30",
} as const

export function VascularReport({ report }: { report: VascularReportT }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2 text-base">
              <Activity className="h-4 w-4 text-primary" aria-hidden="true" />
              Vascular Damage Report
            </CardTitle>
            <CardDescription>LLM-2 · Vision-conditioned</CardDescription>
          </div>
          <span className="inline-flex items-center gap-1 rounded-full border border-border bg-muted px-2 py-0.5 text-[10px] font-mono text-muted-foreground">
            <Sparkles className="h-3 w-3" aria-hidden="true" />
            grounded on biomarkers
          </span>
        </div>
      </CardHeader>
      <CardContent className="space-y-5 text-sm">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              Overall damage score
            </span>
            <span className="font-mono tabular-nums text-2xl font-semibold">
              {report.overall_damage_score}
              <span className="text-sm text-muted-foreground"> / 100</span>
            </span>
          </div>
          <Progress value={report.overall_damage_score} aria-label="Overall damage score" />
        </div>

        <section className="space-y-2">
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            Per-grade severity
          </h3>
          <div className="space-y-1.5">
            {(Object.keys(report.per_grade_score) as Array<keyof typeof report.per_grade_score>).map(
              (g) => {
                const v = report.per_grade_score[g]
                return (
                  <div key={g} className="flex items-center gap-3 text-xs">
                    <span className="w-12 font-mono text-muted-foreground">Grade {g}</span>
                    <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                      <div
                        className="h-full bg-primary transition-all"
                        style={{ width: `${v}%` }}
                        aria-hidden="true"
                      />
                    </div>
                    <span className="w-10 text-right font-mono tabular-nums">{v}%</span>
                  </div>
                )
              },
            )}
          </div>
        </section>

        <section className="space-y-2">
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            Damaged regions
          </h3>
          {report.damaged_regions.length === 0 ? (
            <p className="rounded-md border border-dashed border-border bg-muted/30 px-3 py-4 text-center text-xs text-muted-foreground">
              <Eye className="mx-auto mb-1 h-4 w-4" aria-hidden="true" />
              No focal vascular damage detected.
            </p>
          ) : (
            <ul className="space-y-2">
              {report.damaged_regions.map((r, i) => (
                <li
                  key={i}
                  className="flex items-start gap-3 rounded-md border border-border bg-card p-3"
                >
                  <span className="mt-0.5 inline-flex h-7 w-7 items-center justify-center rounded-md bg-muted text-muted-foreground">
                    <MapPin className="h-3.5 w-3.5" aria-hidden="true" />
                  </span>
                  <div className="flex-1 space-y-1">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-medium capitalize">{r.quadrant.replace("-", " ")}</span>
                      <span
                        className={cn(
                          "rounded-full px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider ring-1",
                          SEVERITY_TONE[r.severity],
                        )}
                      >
                        {r.severity}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground leading-relaxed">{r.finding}</p>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="space-y-2 border-t border-border pt-4">
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            Rationale
          </h3>
          <p className="text-xs text-muted-foreground leading-relaxed text-pretty">
            {report.rationale}
          </p>
        </section>

        {report.needs_specialist_review ? (
          <p className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
            Cross-check flagged disagreement between visual analysis and CNN grade — specialist review recommended.
          </p>
        ) : null}
      </CardContent>
    </Card>
  )
}
