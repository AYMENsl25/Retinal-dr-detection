import { CalendarClock, Heart, ShieldAlert, Sparkles, Stethoscope } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import type { ClinicalReport as ClinicalReportT } from "@/lib/types"

export function ClinicalReport({ report }: { report: ClinicalReportT }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2 text-base">
              <Stethoscope className="h-4 w-4 text-primary" aria-hidden="true" />
              Clinical Report
            </CardTitle>
            <CardDescription>LLM-1 · Reasoning model</CardDescription>
          </div>
          <span className="inline-flex items-center gap-1 rounded-full border border-border bg-muted px-2 py-0.5 text-[10px] font-mono text-muted-foreground">
            <Sparkles className="h-3 w-3" aria-hidden="true" />
            ICDR · AAO PPP
          </span>
        </div>
      </CardHeader>
      <CardContent className="space-y-5 text-sm">
        <Section title="Summary">
          <p className="leading-relaxed text-pretty">{report.summary}</p>
        </Section>

        <Section title="Pathophysiology">
          <p className="leading-relaxed text-pretty text-muted-foreground">{report.pathophysiology}</p>
        </Section>

        <div className="grid gap-5 sm:grid-cols-2">
          <Section title="Recommendations">
            <ul className="space-y-1.5">
              {report.recommendations.map((r, i) => (
                <li key={i} className="flex gap-2 leading-relaxed">
                  <span aria-hidden="true" className="mt-1.5 inline-block h-1 w-1 shrink-0 rounded-full bg-primary" />
                  <span>{r}</span>
                </li>
              ))}
            </ul>
          </Section>

          <Section title="Lifestyle advice">
            <ul className="space-y-1.5">
              {report.lifestyle_advice.map((r, i) => (
                <li key={i} className="flex gap-2 leading-relaxed">
                  <Heart className="h-3 w-3 mt-1 shrink-0 text-primary" aria-hidden="true" />
                  <span>{r}</span>
                </li>
              ))}
            </ul>
          </Section>
        </div>

        <div className="grid gap-5 sm:grid-cols-2">
          <Section title="Follow-up">
            <p className="flex items-center gap-2 leading-relaxed">
              <CalendarClock className="h-4 w-4 text-primary" aria-hidden="true" />
              <span className="font-medium">{report.follow_up_window}</span>
            </p>
          </Section>

          <Section title="Risk factors">
            <ul className="flex flex-wrap gap-1.5">
              {report.risk_factors.map((r, i) => (
                <li
                  key={i}
                  className="rounded-full border border-border bg-muted px-2 py-0.5 text-[11px] text-muted-foreground"
                >
                  {r}
                </li>
              ))}
            </ul>
          </Section>
        </div>

        <Section title="Red flags">
          <ul className="space-y-1.5">
            {report.red_flags.map((r, i) => (
              <li key={i} className="flex gap-2 leading-relaxed">
                <ShieldAlert className="h-3.5 w-3.5 mt-0.5 shrink-0 text-destructive" aria-hidden="true" />
                <span>{r}</span>
              </li>
            ))}
          </ul>
        </Section>

        <p className="rounded-md border border-border bg-muted/40 px-3 py-2 text-[11px] text-muted-foreground">
          {report.disclaimer}
        </p>
      </CardContent>
    </Card>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-1.5">
      <h3 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
        {title}
      </h3>
      <div className="text-sm">{children}</div>
    </section>
  )
}
