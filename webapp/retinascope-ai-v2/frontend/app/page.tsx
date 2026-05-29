import Link from "next/link"
import {
  ArrowRight,
  Activity,
  Brain,
  Eye,
  GitBranch,
  MessageSquare,
  ShieldCheck,
  Sparkles,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { SiteHeader } from "@/components/site-header"

const FEATURES = [
  {
    icon: Eye,
    title: "5-panel dual-mask viewer",
    desc: "Original fundus, probability heatmap, clean binary vessel mask, LLM-annotated damage mask, and red overlay — all derived from a single image.",
  },
  {
    icon: Brain,
    title: "Dual-LLM clinical reasoning",
    desc: "LLM-1 produces an ICDR-grounded clinical narrative. LLM-2 (vision-conditioned) localizes vascular damage with quantitative biomarkers.",
  },
  {
    icon: GitBranch,
    title: "Calibrated confidence",
    desc: "Temperature-scaled softmax, MC-Dropout uncertainty, closeness-to-next-grade, and an automatic specialist-referral flag.",
  },
  {
    icon: MessageSquare,
    title: "Case-scoped consultation",
    desc: "Streaming follow-up Q&A grounded in the current patient's structured report. No off-topic medical advice.",
  },
]

const METRICS = [
  { value: "0.85", label: "Vessel Dice (U-Net ensemble)" },
  { value: "0.96", label: "Grade QWK (CNN grader)" },
  { value: "5", label: "Visualization panels" },
  { value: "2 + chat", label: "LLM reasoning stages" },
]

const PIPELINE = [
  { step: "1", label: "Preprocess", detail: "CLAHE · ROI crop · normalize" },
  { step: "2", label: "Segment", detail: "U-Net → binary vessel mask" },
  { step: "3", label: "Grade", detail: "CNN → calibrated softmax" },
  { step: "4", label: "Analyze", detail: "Skeleton → tortuosity → LLM-2" },
  { step: "5", label: "Report", detail: "LLM-1 clinical narrative" },
]

export default function HomePage() {
  return (
    <>
      <SiteHeader />
      <main>
        {/* Hero */}
        <section className="relative overflow-hidden border-b border-border">
          <div
            className="absolute inset-0 -z-10 opacity-[0.04]"
            style={{
              backgroundImage:
                "radial-gradient(circle at 1px 1px, currentColor 1px, transparent 0)",
              backgroundSize: "24px 24px",
              color: "var(--primary)",
            }}
            aria-hidden="true"
          />
          <div className="mx-auto max-w-7xl px-4 pb-20 pt-16 sm:px-6 lg:px-8 lg:pt-24">
            <div className="grid items-center gap-12 lg:grid-cols-[1.1fr_1fr]">
              <div>
                <Badge
                  variant="secondary"
                  className="mb-5 bg-accent text-accent-foreground border-0"
                >
                  <Sparkles className="mr-1.5 h-3 w-3" aria-hidden="true" />
                  v2.0 · Dual-LLM clinical reasoning
                </Badge>
                <h1 className="text-balance text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
                  Diabetic retinopathy screening,{" "}
                  <span className="text-primary">explained.</span>
                </h1>
                <p className="text-pretty mt-6 max-w-xl text-base leading-relaxed text-muted-foreground sm:text-lg">
                  A clinical-decision-support web application that turns a
                  single retinal fundus image into a vessel segmentation, an
                  ICDR grade with calibrated confidence, and a structured
                  clinician-ready report — backed by dual-LLM reasoning and
                  vascular damage localization.
                </p>

                <div className="mt-8 flex flex-col gap-3 sm:flex-row">
                  <Button asChild size="lg">
                    <Link href="/analyze">
                      Start a new analysis
                      <ArrowRight aria-hidden="true" />
                    </Link>
                  </Button>
                  <Button asChild size="lg" variant="outline">
                    <Link href="#pipeline">View the pipeline</Link>
                  </Button>
                </div>

                <p className="mt-6 flex items-center gap-2 text-xs text-muted-foreground">
                  <ShieldCheck
                    className="h-3.5 w-3.5 text-primary"
                    aria-hidden="true"
                  />
                  Decision support only. Not a medical diagnosis. All findings
                  must be confirmed by a licensed ophthalmologist.
                </p>

                <dl className="mt-12 grid grid-cols-2 gap-x-8 gap-y-6 sm:grid-cols-4">
                  {METRICS.map((m) => (
                    <div key={m.label}>
                      <dt className="text-xs text-muted-foreground">
                        {m.label}
                      </dt>
                      <dd className="mt-1 font-mono text-2xl font-bold tracking-tight text-foreground sm:text-3xl">
                        {m.value}
                      </dd>
                    </div>
                  ))}
                </dl>
              </div>

              {/* Hero visual: mock fundus panel preview */}
              <div className="relative">
                <div className="rounded-2xl border border-border bg-card p-3 shadow-sm">
                  <div className="grid grid-cols-5 gap-2">
                    {[
                      { label: "Original", bg: "from-amber-900 to-amber-700" },
                      { label: "Heatmap", bg: "from-blue-600 to-rose-500" },
                      { label: "Mask", bg: "from-slate-900 to-slate-900" },
                      { label: "Damage", bg: "from-slate-900 to-slate-900" },
                      { label: "Overlay", bg: "from-rose-700 to-orange-600" },
                    ].map((p, i) => (
                      <div key={p.label} className="space-y-1.5">
                        <div
                          className={`relative aspect-square overflow-hidden rounded-md bg-gradient-to-br ${p.bg}`}
                        >
                          <div
                            className="absolute inset-0 opacity-50"
                            style={{
                              backgroundImage:
                                "radial-gradient(circle at center, transparent 30%, rgba(0,0,0,0.6) 70%)",
                            }}
                          />
                          {(i === 2 || i === 3) && (
                            <svg
                              viewBox="0 0 64 64"
                              className="absolute inset-0 h-full w-full"
                              aria-hidden="true"
                            >
                              <g
                                fill="none"
                                stroke="white"
                                strokeWidth="0.8"
                                strokeLinecap="round"
                              >
                                <path d="M32 8 Q 26 24, 18 32 T 8 50" />
                                <path d="M32 8 Q 38 22, 46 30 T 56 48" />
                                <path d="M14 20 Q 22 30, 30 38 T 44 52" />
                                <path d="M50 20 Q 42 30, 34 36 T 22 50" />
                                <path d="M20 12 Q 28 24, 36 30 T 50 44" />
                              </g>
                              {i === 3 && (
                                <>
                                  <ellipse
                                    cx="22"
                                    cy="20"
                                    rx="6"
                                    ry="3.5"
                                    fill="none"
                                    stroke="#ef4444"
                                    strokeWidth="1"
                                  />
                                  <ellipse
                                    cx="44"
                                    cy="38"
                                    rx="5"
                                    ry="3"
                                    fill="none"
                                    stroke="#f59e0b"
                                    strokeWidth="0.8"
                                  />
                                </>
                              )}
                            </svg>
                          )}
                        </div>
                        <div className="truncate text-[9px] font-medium text-muted-foreground">
                          {p.label}
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="mt-3 rounded-md border border-border bg-secondary/30 p-3">
                    <div className="flex items-baseline justify-between">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-2xl font-bold text-primary">
                          0
                        </span>
                        <span className="text-sm font-medium">No DR</span>
                      </div>
                      <span className="font-mono text-xs text-muted-foreground">
                        75% conf · 17% → 1
                      </span>
                    </div>
                    <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-secondary">
                      <div
                        className="h-full bg-primary"
                        style={{ width: "75%" }}
                      />
                    </div>
                  </div>
                </div>
                <div
                  className="absolute -bottom-4 -right-4 -z-10 h-32 w-32 rounded-full bg-primary/10 blur-3xl"
                  aria-hidden="true"
                />
              </div>
            </div>
          </div>
        </section>

        {/* Features */}
        <section className="border-b border-border py-20">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="mb-12 max-w-2xl">
              <h2 className="text-balance text-3xl font-bold tracking-tight sm:text-4xl">
                Built for the ophthalmology workflow
              </h2>
              <p className="text-pretty mt-4 text-muted-foreground">
                Every output is grounded, calibrated, and traceable. Vessel
                segmentation, ICDR grading, and LLM reasoning combine into one
                clinician-ready report.
              </p>
            </div>
            <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
              {FEATURES.map((f) => (
                <Card key={f.title} className="border-border">
                  <CardContent className="pt-6">
                    <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-accent text-accent-foreground">
                      <f.icon className="h-5 w-5" aria-hidden="true" />
                    </div>
                    <h3 className="mb-2 text-base font-semibold">{f.title}</h3>
                    <p className="text-sm leading-relaxed text-muted-foreground">
                      {f.desc}
                    </p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </section>

        {/* Pipeline */}
        <section id="pipeline" className="border-b border-border py-20">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="mb-12">
              <Badge
                variant="secondary"
                className="mb-3 bg-accent text-accent-foreground border-0"
              >
                Pipeline
              </Badge>
              <h2 className="text-balance text-3xl font-bold tracking-tight sm:text-4xl">
                A single image, five rigorous stages
              </h2>
            </div>
            <ol className="grid gap-4 md:grid-cols-5">
              {PIPELINE.map((p) => (
                <li
                  key={p.step}
                  className="rounded-xl border border-border bg-card p-5"
                >
                  <div className="mb-3 flex items-center gap-2">
                    <span className="flex h-7 w-7 items-center justify-center rounded-md bg-primary font-mono text-xs font-bold text-primary-foreground">
                      {p.step}
                    </span>
                    <span className="font-semibold">{p.label}</span>
                  </div>
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    {p.detail}
                  </p>
                </li>
              ))}
            </ol>
          </div>
        </section>

        {/* Disclaimer / CTA */}
        <section className="py-20">
          <div className="mx-auto max-w-3xl px-4 text-center sm:px-6 lg:px-8">
            <Activity
              className="mx-auto mb-6 h-10 w-10 text-primary"
              aria-hidden="true"
            />
            <h2 className="text-balance text-3xl font-bold tracking-tight sm:text-4xl">
              Ready to analyze a fundus image?
            </h2>
            <p className="text-pretty mt-4 text-muted-foreground">
              Drop a single color fundus photograph to receive a full
              decision-support report. All findings remain the responsibility of
              the reviewing clinician.
            </p>
            <div className="mt-8">
              <Button asChild size="lg">
                <Link href="/analyze">
                  Open the analyzer
                  <ArrowRight aria-hidden="true" />
                </Link>
              </Button>
            </div>
          </div>
        </section>

        <footer className="border-t border-border py-8">
          <div className="mx-auto max-w-7xl px-4 text-center text-xs text-muted-foreground sm:px-6 lg:px-8">
            <p>
              RetinaScope-AI — Decision support only. Not a medical diagnosis.
              All findings must be confirmed by a licensed ophthalmologist.
            </p>
            <p className="mt-1">
              © {new Date().getFullYear()} RetinaScope-AI · Research &amp;
              clinical decision support
            </p>
          </div>
        </footer>
      </main>
    </>
  )
}
