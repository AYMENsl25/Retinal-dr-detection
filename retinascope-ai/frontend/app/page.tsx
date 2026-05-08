import Link from "next/link"
import { Activity, ArrowRight, Brain, Eye, GitBranch, MessageSquare, ShieldCheck, Stethoscope } from "lucide-react"
import { SiteHeader } from "@/components/site-header"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"

const FEATURES = [
  {
    icon: GitBranch,
    title: "4-panel viewer",
    desc: "Original · Binary mask · Probability heatmap · Red overlay — pixel-aligned to the source image.",
  },
  {
    icon: Activity,
    title: "Calibrated grading",
    desc: "Temperature-scaled softmax across the 5-step ICDR scale, with closeness-to-next-grade scoring.",
  },
  {
    icon: Stethoscope,
    title: "Clinical reasoning",
    desc: "LLM-1 produces a structured ICDR/AAO-grounded narrative with follow-up window and red flags.",
  },
  {
    icon: Brain,
    title: "Vascular damage analyst",
    desc: "Vision-conditioned LLM-2 cross-checks the CNN grade against quantitative vessel biomarkers.",
  },
  {
    icon: MessageSquare,
    title: "Consultation chat",
    desc: "Streaming follow-up Q&A scoped to the active case context — never loses the patient.",
  },
  {
    icon: ShieldCheck,
    title: "Decision support, not diagnosis",
    desc: "Explicit disclaimers, JSON-locked LLM outputs, and an auto-refer trigger on uncertainty.",
  },
]

const STAGES = [
  { n: "01", title: "Preprocess", desc: "Crop to FOV · CLAHE · resize · normalize" },
  { n: "02", title: "Segment", desc: "Vessel U-Net · Lesion U-Net" },
  { n: "03", title: "Grade", desc: "CNN classifier · temp-scaled softmax · MC-Dropout" },
  { n: "04", title: "Reason", desc: "LLM-1 (clinical) · LLM-2 (vascular, vision)" },
]

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background">
      <SiteHeader />

      <main>
        {/* Hero */}
        <section className="relative overflow-hidden border-b border-border">
          <div
            aria-hidden="true"
            className="absolute inset-0 -z-10 bg-[radial-gradient(ellipse_at_top,theme(colors.primary/12%),transparent_60%)]"
          />
          <div className="mx-auto max-w-7xl px-4 md:px-6 py-16 md:py-24">
            <div className="max-w-3xl space-y-6">
              <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-card px-3 py-1 text-xs font-medium text-muted-foreground">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-75" />
                  <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-primary" />
                </span>
                v0.1 · mock pipeline · drop-in your .pth checkpoints
              </span>
              <h1 className="text-4xl md:text-6xl font-bold tracking-tight text-balance">
                Diabetic retinopathy screening,{" "}
                <span className="text-primary">explained.</span>
              </h1>
              <p className="text-lg text-muted-foreground text-pretty max-w-2xl leading-relaxed">
                A clinical-decision-support web app that turns a single retinal fundus image into a
                vessel segmentation, an ICDR grade, calibrated confidence, and a structured
                clinician-ready report — backed by dual-LLM reasoning.
              </p>
              <div className="flex flex-wrap items-center gap-3">
                <Button asChild size="lg">
                  <Link href="/analyze">
                    Try the demo
                    <ArrowRight className="h-4 w-4" />
                  </Link>
                </Button>
                <Button asChild size="lg" variant="outline">
                  <Link href="#pipeline">View the pipeline</Link>
                </Button>
              </div>

              <dl className="grid grid-cols-3 gap-6 max-w-lg pt-6">
                <Stat label="Vessel Dice" value="0.87" />
                <Stat label="Grade QWK" value="0.96" />
                <Stat label="LLM stages" value="2 + chat" />
              </dl>
            </div>
          </div>
        </section>

        {/* Pipeline */}
        <section id="pipeline" className="border-b border-border">
          <div className="mx-auto max-w-7xl px-4 md:px-6 py-16 space-y-10">
            <div className="max-w-2xl space-y-2">
              <p className="text-xs font-semibold uppercase tracking-wider text-primary">
                Request lifecycle
              </p>
              <h2 className="text-3xl font-bold tracking-tight">From upload to clinician-ready report</h2>
              <p className="text-muted-foreground text-pretty leading-relaxed">
                Four asynchronous stages, parallelized where possible. Each step exposes a typed
                contract so you can swap in your own model checkpoints without touching the UI.
              </p>
            </div>
            <ol className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              {STAGES.map((s) => (
                <Card key={s.n} className="relative overflow-hidden">
                  <CardContent className="p-5 space-y-2">
                    <span className="text-xs font-mono text-muted-foreground">{s.n}</span>
                    <h3 className="font-semibold tracking-tight">{s.title}</h3>
                    <p className="text-xs text-muted-foreground leading-relaxed">{s.desc}</p>
                  </CardContent>
                  <span
                    aria-hidden="true"
                    className="absolute right-3 top-3 inline-flex h-6 w-6 items-center justify-center rounded-md bg-primary/10 text-primary"
                  >
                    <Eye className="h-3 w-3" />
                  </span>
                </Card>
              ))}
            </ol>
          </div>
        </section>

        {/* Features */}
        <section className="border-b border-border">
          <div className="mx-auto max-w-7xl px-4 md:px-6 py-16 space-y-10">
            <div className="max-w-2xl space-y-2">
              <p className="text-xs font-semibold uppercase tracking-wider text-primary">Capabilities</p>
              <h2 className="text-3xl font-bold tracking-tight">Built for the way clinicians read images</h2>
            </div>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {FEATURES.map((f) => {
                const Icon = f.icon
                return (
                  <Card key={f.title}>
                    <CardContent className="p-5 space-y-3">
                      <span className="inline-flex h-9 w-9 items-center justify-center rounded-md bg-primary/10 text-primary">
                        <Icon className="h-4 w-4" aria-hidden="true" />
                      </span>
                      <h3 className="font-semibold tracking-tight">{f.title}</h3>
                      <p className="text-sm text-muted-foreground leading-relaxed text-pretty">
                        {f.desc}
                      </p>
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          </div>
        </section>

        {/* CTA */}
        <section>
          <div className="mx-auto max-w-7xl px-4 md:px-6 py-16">
            <Card className="overflow-hidden">
              <CardContent className="flex flex-col gap-6 p-8 sm:flex-row sm:items-center sm:justify-between">
                <div className="space-y-1.5 max-w-xl">
                  <h2 className="text-2xl font-bold tracking-tight text-balance">
                    Drop in your checkpoints, ship a clinician-ready demo.
                  </h2>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    The mock backend implements the exact <code className="font-mono text-xs">/api/v1/analyze</code> contract from the project plan — replace it with your FastAPI service and the UI keeps working.
                  </p>
                </div>
                <Button asChild size="lg">
                  <Link href="/analyze">
                    Start screening
                    <ArrowRight className="h-4 w-4" />
                  </Link>
                </Button>
              </CardContent>
            </Card>
          </div>
        </section>
      </main>

      <footer className="border-t border-border">
        <div className="mx-auto max-w-7xl px-4 md:px-6 py-6 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between text-xs text-muted-foreground">
          <p>RetinaScope-AI · Decision support only — not a medical diagnosis.</p>
          <p>Built on the §1–13 plan · v0.app</p>
        </div>
      </footer>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="space-y-0.5">
      <dd className="text-2xl font-bold tabular-nums">{value}</dd>
      <dt className="text-xs text-muted-foreground">{label}</dt>
    </div>
  )
}
