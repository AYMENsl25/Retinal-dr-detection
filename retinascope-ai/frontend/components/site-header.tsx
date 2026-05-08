"use client"

import Link from "next/link"
import { Download, Eye } from "lucide-react"
import { Button } from "@/components/ui/button"

interface SiteHeaderProps {
  patientId?: string
  onExportPdf?: () => void
}

export function SiteHeader({ patientId, onExportPdf }: SiteHeaderProps) {
  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-30 print:hidden">
      <div className="mx-auto max-w-7xl flex items-center justify-between px-4 md:px-6 h-14">
        <Link href="/" className="flex items-center gap-2">
          <span className="flex h-8 w-8 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <Eye className="h-4 w-4" aria-hidden="true" />
          </span>
          <span className="font-semibold tracking-tight">RetinaScope-AI</span>
          <span className="hidden sm:inline rounded-full border border-border bg-muted px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
            Decision Support
          </span>
        </Link>

        <div className="flex items-center gap-3">
          {patientId ? (
            <span className="hidden md:inline text-xs text-muted-foreground">
              Case <span className="font-mono text-foreground">{patientId.slice(0, 8)}</span>
            </span>
          ) : null}
          <Button variant="ghost" size="sm" asChild>
            <Link href="/analyze">Analyze</Link>
          </Button>
          <Button
            size="sm"
            variant="outline"
            disabled={!onExportPdf}
            aria-disabled={!onExportPdf}
            onClick={onExportPdf}
            title={onExportPdf ? "Open print dialog to save as PDF" : "Run an analysis before exporting"}
          >
            <Download className="h-3.5 w-3.5" aria-hidden="true" />
            Export PDF
          </Button>
        </div>
      </div>
    </header>
  )
}
