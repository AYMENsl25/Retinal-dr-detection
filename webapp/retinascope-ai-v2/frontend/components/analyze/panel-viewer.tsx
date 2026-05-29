"use client"

import { useState } from "react"
import { Maximize2, X } from "lucide-react"
import { Slider } from "@/components/ui/slider"
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { Panels } from "@/lib/types"

interface PanelDef {
  key: keyof Pick<
    Panels,
    "original" | "heatmap" | "cleanMask" | "dmgMask" | "overlay"
  >
  label: string
  sub: string
}

const PANELS: PanelDef[] = [
  { key: "original", label: "Original Fundus", sub: "CLAHE preprocessed" },
  { key: "heatmap", label: "Probability Heatmap", sub: "vessel softmax · jet" },
  { key: "cleanMask", label: "Binary Vessel Mask", sub: "U-Net @ τ = 0.5" },
  { key: "dmgMask", label: "Damage Analysis", sub: "LLM-annotated regions" },
  { key: "overlay", label: "Vessel Overlay", sub: "mask × original" },
]

interface PanelViewerProps {
  panels: Panels
}

export function PanelViewer({ panels }: PanelViewerProps) {
  const [opacity, setOpacity] = useState(75)
  const [zoom, setZoom] = useState<PanelDef | null>(null)

  return (
    <div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        {PANELS.map((p, i) => (
          <button
            key={p.key}
            type="button"
            className="group text-left focus:outline-none"
            onClick={() => setZoom(p)}
            aria-label={`Zoom panel ${p.label}`}
          >
            <div className="relative overflow-hidden rounded-lg border border-border bg-black ring-offset-background transition-all group-hover:ring-2 group-hover:ring-primary group-focus-visible:ring-2 group-focus-visible:ring-primary">
              <img
                src={panels[p.key] || "/placeholder.svg"}
                alt={p.label}
                className="block aspect-square w-full object-cover"
              />
              {p.key === "overlay" && (
                <img
                  src={panels.original || "/placeholder.svg"}
                  alt=""
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-0 aspect-square w-full object-cover"
                  style={{ opacity: 1 - opacity / 100 }}
                />
              )}
              <div className="pointer-events-none absolute right-1.5 top-1.5 rounded bg-black/50 p-1 opacity-0 transition-opacity group-hover:opacity-100">
                <Maximize2
                  className="h-3 w-3 text-white"
                  aria-hidden="true"
                />
              </div>
              <div className="pointer-events-none absolute left-1.5 top-1.5 rounded bg-black/50 px-1.5 py-0.5 font-mono text-[10px] font-medium text-white">
                {i + 1}
              </div>
            </div>
            <div className="mt-1.5 text-xs font-medium text-foreground">
              {p.label}
            </div>
            <div className="text-[10px] text-muted-foreground">{p.sub}</div>
          </button>
        ))}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3 rounded-lg border border-border bg-secondary/30 p-3">
        <span className="text-xs font-medium text-muted-foreground">
          Overlay opacity (panel 5)
        </span>
        <div className="flex-1 min-w-[140px] max-w-xs">
          <Slider
            value={[opacity]}
            onValueChange={(v) => setOpacity(v[0])}
            min={0}
            max={100}
            step={1}
            aria-label="Overlay opacity"
          />
        </div>
        <span className="font-mono text-xs tabular-nums text-foreground">
          {opacity}%
        </span>
        <span className="text-[10px] text-muted-foreground sm:ml-auto">
          Vessel Dice 0.87 · Grader QWK 0.96 (reported)
        </span>
      </div>

      <Dialog open={!!zoom} onOpenChange={(o) => !o && setZoom(null)}>
        <DialogContent className="max-w-3xl border-border p-0">
          <DialogTitle className="sr-only">
            {zoom?.label} — zoomed view
          </DialogTitle>
          {zoom && (
            <div className="relative">
              <img
                src={panels[zoom.key] || "/placeholder.svg"}
                alt={zoom.label}
                className="block w-full rounded-t-lg bg-black"
              />
              <div className="flex items-start justify-between gap-4 p-4">
                <div>
                  <div className="text-sm font-semibold">{zoom.label}</div>
                  <div className="text-xs text-muted-foreground">
                    {zoom.sub}
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setZoom(null)}
                  aria-label="Close zoom"
                >
                  <X className="h-4 w-4" aria-hidden="true" />
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}

export { PANELS as PANEL_DEFS }
export { cn }
