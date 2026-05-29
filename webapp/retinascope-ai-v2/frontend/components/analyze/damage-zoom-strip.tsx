"use client"

import { useState } from "react"
import { ChevronDown, ChevronUp } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog"
import { cn } from "@/lib/utils"
import type { Severity, ZoomCrop } from "@/lib/types"

interface DamageZoomStripProps {
  crops: ZoomCrop[]
}

const SEVERITY_STYLES: Record<Severity, string> = {
  high: "border-destructive/60 bg-destructive/5",
  medium: "border-warning/60 bg-warning/5",
  low: "border-success/60 bg-success/5",
}

const SEVERITY_BADGE: Record<Severity, string> = {
  high: "bg-destructive/15 text-destructive border-destructive/30",
  medium: "bg-warning/15 text-warning-foreground border-warning/30",
  low: "bg-success/15 text-success border-success/30",
}

export function DamageZoomStrip({ crops }: DamageZoomStripProps) {
  const [expanded, setExpanded] = useState(true)
  const [modal, setModal] = useState<ZoomCrop | null>(null)

  if (!crops?.length) return null

  return (
    <div className="mt-4 rounded-lg border border-border bg-card">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center justify-between gap-2 px-4 py-3 text-left"
        aria-expanded={expanded}
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold">Damage zoom crops</span>
          <Badge variant="outline" className="font-mono text-[10px]">
            {crops.length} region{crops.length > 1 ? "s" : ""}
          </Badge>
        </div>
        {expanded ? (
          <ChevronUp className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
        ) : (
          <ChevronDown
            className="h-4 w-4 text-muted-foreground"
            aria-hidden="true"
          />
        )}
      </button>

      {expanded && (
        <div className="border-t border-border p-4">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            {crops.map((c, i) => (
              <button
                key={i}
                type="button"
                onClick={() => setModal(c)}
                className={cn(
                  "group overflow-hidden rounded-lg border-2 text-left transition-transform hover:scale-[1.02] focus:outline-none focus-visible:ring-2 focus-visible:ring-primary",
                  SEVERITY_STYLES[c.severity],
                )}
              >
                <img
                  src={c.src || "/placeholder.svg"}
                  alt={c.finding}
                  className="block aspect-square w-full bg-black object-cover"
                />
                <div className="space-y-1 p-2">
                  <div className="flex items-center justify-between gap-1">
                    <span
                      className={cn(
                        "rounded border px-1.5 py-0.5 font-mono text-[9px] font-bold uppercase",
                        SEVERITY_BADGE[c.severity],
                      )}
                    >
                      {c.severity}
                    </span>
                    <span className="font-mono text-[10px] text-muted-foreground">
                      {c.quadrant}
                    </span>
                  </div>
                  <div className="line-clamp-2 text-[10px] leading-snug text-muted-foreground">
                    {c.finding}
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      <Dialog open={!!modal} onOpenChange={(o) => !o && setModal(null)}>
        <DialogContent className="max-w-sm">
          <DialogTitle className="sr-only">Damage region detail</DialogTitle>
          {modal && (
            <div>
              <img
                src={modal.src || "/placeholder.svg"}
                alt={modal.finding}
                className="block w-full rounded-md bg-black"
              />
              <div className="mt-4 space-y-3">
                <div className="flex items-center gap-2">
                  <span
                    className={cn(
                      "rounded border px-1.5 py-0.5 font-mono text-[10px] font-bold uppercase",
                      SEVERITY_BADGE[modal.severity],
                    )}
                  >
                    {modal.severity} severity
                  </span>
                  <span className="font-mono text-xs text-muted-foreground">
                    Quadrant {modal.quadrant}
                  </span>
                </div>
                <p className="text-sm font-medium">{modal.finding}</p>
                <p className="text-xs text-muted-foreground">
                  Identified by LLM-2 vascular damage analyst, grounded in
                  skeleton and connectivity metrics. Confirm visually before
                  acting.
                </p>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}
