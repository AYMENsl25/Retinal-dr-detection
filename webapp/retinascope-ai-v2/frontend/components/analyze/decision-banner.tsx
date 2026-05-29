"use client"

import { AlertTriangle, AlertCircle, CheckCircle2 } from "lucide-react"
import type { DecisionFlag } from "@/lib/types"

interface DecisionBannerProps {
  flag: DecisionFlag
  entropy: number
}

export function DecisionBanner({ flag, entropy }: DecisionBannerProps) {
  if (flag === "HIGH_CONCERN") {
    return (
      <div
        role="alert"
        className="mb-5 flex items-center gap-3 rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3"
      >
        <AlertTriangle
          className="h-5 w-5 shrink-0 text-destructive"
          aria-hidden="true"
        />
        <div className="flex-1">
          <div className="text-sm font-semibold text-destructive">
            HIGH CONCERN — Specialist referral recommended
          </div>
          <div className="text-xs text-destructive/80">
            Combined CNN grade and vascular analysis indicate referable disease.
            Confirm with a licensed ophthalmologist.
          </div>
        </div>
      </div>
    )
  }

  if (flag === "MEDIUM") {
    return (
      <div
        role="status"
        className="mb-5 flex items-center gap-3 rounded-lg border border-warning/40 bg-warning/10 px-4 py-3"
      >
        <AlertCircle
          className="h-5 w-5 shrink-0 text-warning"
          aria-hidden="true"
        />
        <div className="flex-1">
          <div className="text-sm font-semibold text-warning-foreground">
            MEDIUM — Second look advised
          </div>
          <div className="text-xs text-muted-foreground">
            Uncertainty entropy {entropy.toFixed(2)} or borderline grade.
            Recommend manual review.
          </div>
        </div>
      </div>
    )
  }

  return (
    <div
      role="status"
      className="mb-5 flex items-center gap-3 rounded-lg border border-success/40 bg-success/10 px-4 py-3"
    >
      <CheckCircle2
        className="h-5 w-5 shrink-0 text-success"
        aria-hidden="true"
      />
      <div className="flex-1">
        <div className="text-sm font-semibold text-success">
          HIGH CONFIDENCE — No immediate referral indicated
        </div>
        <div className="text-xs text-muted-foreground">
          Entropy {entropy.toFixed(2)}, calibrated grade with strong vascular
          agreement.
        </div>
      </div>
    </div>
  )
}
