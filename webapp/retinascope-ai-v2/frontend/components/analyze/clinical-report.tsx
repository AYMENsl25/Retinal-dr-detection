"use client"

import { Stethoscope, Clock, AlertOctagon, HeartPulse } from "lucide-react"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Separator } from "@/components/ui/separator"
import type { ClinicalReportData } from "@/lib/types"

interface ClinicalReportProps {
  loading: boolean
  report: ClinicalReportData | null
  error?: string | null
}

export function ClinicalReport({ loading, report, error }: ClinicalReportProps) {
  return (
    <Card className="border-border">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-md bg-accent text-accent-foreground">
              <Stethoscope className="h-4 w-4" aria-hidden="true" />
            </span>
            <div>
              <div className="text-sm font-semibold">Clinical Report</div>
              <div className="text-[11px] text-muted-foreground">
                LLM-1 · Reasoning model · ICDR + AAO PPP
              </div>
            </div>
          </div>
          <div className="flex flex-wrap gap-1">
            <Badge variant="secondary" className="text-[10px]">
              ICDR
            </Badge>
            <Badge variant="secondary" className="text-[10px]">
              AAO PPP
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {loading && (
          <div className="space-y-3" aria-busy="true" aria-live="polite">
            <Skeleton className="h-3 w-24" />
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-5/6" />
            <Skeleton className="h-3 w-3/4 mt-4" />
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-2/3" />
          </div>
        )}

        {error && !loading && (
          <div className="rounded-md border border-destructive/30 bg-destructive/5 p-3 text-xs text-destructive">
            {error}
          </div>
        )}

        {report && !loading && (
          <div className="space-y-4 text-sm leading-relaxed">
            <Section title="Summary">
              <p>{report.summary}</p>
            </Section>

            <Section title="Pathophysiology">
              <p className="text-muted-foreground">{report.pathophysiology}</p>
            </Section>

            <div className="grid gap-4 sm:grid-cols-2">
              <Section title="Recommendations">
                <BulletList items={report.recommendations} />
              </Section>
              <Section title="Lifestyle Advice">
                <BulletList items={report.lifestyle_advice} />
              </Section>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <Section title="Risk Factors">
                <BulletList items={report.risk_factors} />
              </Section>
              <Section title="Follow-up">
                <p className="flex items-center gap-2 text-muted-foreground">
                  <Clock className="h-4 w-4 text-primary" aria-hidden="true" />
                  {report.follow_up_window}
                </p>
              </Section>
            </div>

            {report.red_flags?.length > 0 && (
              <div className="rounded-md border border-destructive/30 bg-destructive/5 p-3">
                <div className="mb-2 flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-destructive">
                  <AlertOctagon className="h-3.5 w-3.5" aria-hidden="true" />
                  Red Flags
                </div>
                <BulletList items={report.red_flags} tone="destructive" />
              </div>
            )}

            <Separator />
            <p className="flex items-start gap-1.5 text-[11px] leading-relaxed text-muted-foreground">
              <HeartPulse
                className="mt-0.5 h-3 w-3 shrink-0 text-primary"
                aria-hidden="true"
              />
              {report.disclaimer}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function Section({
  title,
  children,
}: {
  title: string
  children: React.ReactNode
}) {
  return (
    <div>
      <h3 className="mb-1.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
        {title}
      </h3>
      <div className="text-sm">{children}</div>
    </div>
  )
}

function BulletList({
  items,
  tone,
}: {
  items: string[]
  tone?: "destructive"
}) {
  return (
    <ul className="space-y-1.5">
      {items.map((it, i) => (
        <li key={i} className="flex gap-2 text-sm leading-relaxed">
          <span
            className={
              tone === "destructive"
                ? "mt-1.5 h-1 w-1 shrink-0 rounded-full bg-destructive"
                : "mt-1.5 h-1 w-1 shrink-0 rounded-full bg-primary"
            }
            aria-hidden="true"
          />
          <span
            className={
              tone === "destructive" ? "text-destructive" : "text-muted-foreground"
            }
          >
            {it}
          </span>
        </li>
      ))}
    </ul>
  )
}
