import type { Metadata } from "next"
import Link from "next/link"
import {
  ArrowRight,
  ShieldCheck,
  BookOpen,
  AlertTriangle,
  Microscope,
} from "lucide-react"
import { SiteHeader } from "@/components/site-header"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"

export const metadata: Metadata = {
  title: "About · RetinaScope-AI",
  description:
    "How RetinaScope-AI works, what it is for, and the responsibilities of the reviewing clinician.",
}

export default function AboutPage() {
  return (
    <>
      <SiteHeader />
      <main className="mx-auto max-w-3xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="mb-10">
          <h1 className="text-balance text-4xl font-bold tracking-tight">
            About RetinaScope-AI
          </h1>
          <p className="text-pretty mt-4 text-muted-foreground">
            RetinaScope-AI is a clinical-decision-support web application
            developed to assist ophthalmology workflows in screening for
            diabetic retinopathy (DR). It combines two trained deep-learning
            models — a U-Net vessel segmenter and a CNN ICDR grader — with
            quantitative skeletal-vessel analysis and dual-LLM clinical
            reasoning.
          </p>
        </div>

        <Section
          icon={Microscope}
          title="What the system does"
        >
          <ul className="space-y-2 text-sm leading-relaxed text-muted-foreground">
            <li>
              Produces a clean binary vessel segmentation mask (U-Net ensemble,
              reported Dice 0.87).
            </li>
            <li>
              Outputs an ICDR DR grade (0–4) with temperature-scaled,
              calibrated softmax probabilities (reported QWK 0.96).
            </li>
            <li>
              Computes vascular biomarkers — tortuosity, fractal dimension, AVR,
              vessel density, broken-segment count, quadrant density.
            </li>
            <li>
              Localizes likely vessel-damage regions via a vision-conditioned
              LLM that returns bounding boxes the backend renders as red
              ellipses on the mask.
            </li>
            <li>
              Generates a structured clinical narrative grounded in ICDR and the
              AAO Preferred Practice Pattern.
            </li>
            <li>
              Provides a case-scoped streaming consultation chat for follow-up
              questions.
            </li>
          </ul>
        </Section>

        <Section icon={BookOpen} title="Intended use">
          <p className="text-sm leading-relaxed text-muted-foreground">
            RetinaScope-AI is intended as <strong>decision support</strong> for
            licensed eye-care professionals (ophthalmologists, optometrists)
            screening adult diabetic patients. It is{" "}
            <strong>not a diagnostic device</strong>, not a replacement for a
            dilated fundus examination, and not approved for autonomous
            grading. Every result must be reviewed and confirmed by the
            responsible clinician before any clinical action is taken.
          </p>
        </Section>

        <Section icon={ShieldCheck} title="Privacy &amp; data handling">
          <ul className="space-y-2 text-sm leading-relaxed text-muted-foreground">
            <li>
              Uploaded images are processed for inference and may be persisted
              alongside the case record for audit purposes.
            </li>
            <li>
              No image leaves the configured infrastructure perimeter. LLM
              calls go through a controlled gateway with prompt and output
              logging for traceability.
            </li>
            <li>
              Anonymize patient identifiers before upload unless your local
              policy and consent process explicitly permits otherwise.
            </li>
          </ul>
        </Section>

        <Section icon={AlertTriangle} title="Known limitations">
          <ul className="space-y-2 text-sm leading-relaxed text-muted-foreground">
            <li>
              Performance is reported on public DR datasets; domain shift in
              new clinics or cameras may reduce accuracy.
            </li>
            <li>
              The lesion-segmentation model (microaneurysms, hemorrhages, hard
              exudates, optic disc, cotton-wool spots) is intentionally{" "}
              <em>not</em> exposed in this version — see PROJECT_PLAN v2 §1.
            </li>
            <li>
              LLM-generated damage regions are heuristic visual estimates,
              cross-checked against quantitative biomarkers but still
              susceptible to hallucination. The UI surfaces a specialist-review
              flag when biomarkers and grader disagree.
            </li>
            <li>
              Confidence scores are calibrated but not infallible. Use them as
              a triage signal, not a threshold for clinical action.
            </li>
          </ul>
        </Section>

        <div className="mt-10 rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive">
          <strong>Reminder:</strong> Decision support only. Not a medical
          diagnosis. All findings must be confirmed by a licensed
          ophthalmologist.
        </div>

        <div className="mt-10 flex justify-center">
          <Button asChild size="lg">
            <Link href="/analyze">
              Open the analyzer
              <ArrowRight aria-hidden="true" />
            </Link>
          </Button>
        </div>
      </main>
    </>
  )
}

function Section({
  icon: Icon,
  title,
  children,
}: {
  icon: React.ComponentType<{ className?: string; "aria-hidden"?: boolean }>
  title: string
  children: React.ReactNode
}) {
  return (
    <Card className="mb-6 border-border">
      <CardContent className="space-y-3 p-6">
        <div className="flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-md bg-accent text-accent-foreground">
            <Icon className="h-4 w-4" aria-hidden />
          </span>
          <h2 className="text-lg font-semibold">{title}</h2>
        </div>
        {children}
      </CardContent>
    </Card>
  )
}
