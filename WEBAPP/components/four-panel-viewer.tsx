"use client"

import { useState } from "react"
import type { PanelImages } from "@/lib/image-processing"
import { Slider } from "@/components/ui/slider"
import { Card } from "@/components/ui/card"

interface FourPanelViewerProps {
  panels: PanelImages
}

const PANEL_META: { key: keyof PanelImages; title: string; subtitle: string }[] = [
  { key: "original", title: "Original Fundus", subtitle: "CLAHE-preprocessed input" },
  { key: "mask", title: "Binary Vessel Mask", subtitle: "U-Net @ τ = 0.5" },
  { key: "heatmap", title: "Probability Heatmap", subtitle: "Vessel softmax (jet)" },
  { key: "overlay", title: "Red Overlay", subtitle: "Mask × Original" },
]

export function FourPanelViewer({ panels }: FourPanelViewerProps) {
  const [opacity, setOpacity] = useState(75)

  return (
    <Card className="overflow-hidden">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-px bg-border">
        {PANEL_META.map((panel) => {
          const src = panels[panel.key]
          const isOverlay = panel.key === "overlay"
          return (
            <figure key={panel.key} className="bg-card flex flex-col">
              <div className="relative aspect-square bg-foreground">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={panels.original || "/placeholder.svg"}
                  alt=""
                  aria-hidden="true"
                  className="absolute inset-0 h-full w-full object-cover"
                  style={{ visibility: isOverlay ? "visible" : "hidden" }}
                />
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={src || "/placeholder.svg"}
                  alt={panel.title}
                  className="absolute inset-0 h-full w-full object-cover"
                  style={isOverlay ? { opacity: opacity / 100 } : undefined}
                />
                {panel.key === "heatmap" ? <ColorbarLegend /> : null}
              </div>
              <figcaption className="px-3 py-2 border-t border-border">
                <p className="text-xs font-semibold tracking-tight">{panel.title}</p>
                <p className="text-[11px] text-muted-foreground">{panel.subtitle}</p>
              </figcaption>
            </figure>
          )
        })}
      </div>

      <div className="flex flex-col gap-2 border-t border-border bg-muted/30 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <label
            htmlFor="overlay-opacity"
            className="text-xs font-medium text-muted-foreground whitespace-nowrap"
          >
            Overlay opacity
          </label>
          <Slider
            id="overlay-opacity"
            min={0}
            max={100}
            step={1}
            value={[opacity]}
            onValueChange={(v) => setOpacity(v[0])}
            className="w-full sm:w-56"
            aria-label="Overlay opacity"
          />
          <span className="text-xs font-mono w-9 text-right">{opacity}%</span>
        </div>
        <p className="text-[11px] text-muted-foreground">
          Vessel segmentation · Dice 0.87 (reported)
        </p>
      </div>
    </Card>
  )
}

function ColorbarLegend() {
  return (
    <div className="absolute bottom-2 right-2 flex items-center gap-1 rounded-md bg-background/85 px-1.5 py-1 backdrop-blur-sm">
      <span className="text-[9px] font-mono text-foreground">0</span>
      <span
        aria-hidden="true"
        className="block h-1.5 w-16 rounded-sm"
        style={{
          background:
            "linear-gradient(to right, rgb(0,0,143), rgb(0,255,255), rgb(255,255,0), rgb(255,0,0), rgb(143,0,0))",
        }}
      />
      <span className="text-[9px] font-mono text-foreground">1</span>
    </div>
  )
}
