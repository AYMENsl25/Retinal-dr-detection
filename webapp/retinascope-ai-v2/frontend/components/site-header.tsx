"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Activity, FileDown, ShieldCheck } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

interface SiteHeaderProps {
  caseId?: string | null
  onExport?: () => void
  showExport?: boolean
}

export function SiteHeader({ caseId, onExport, showExport }: SiteHeaderProps) {
  const pathname = usePathname()

  const navItems = [
    { href: "/analyze", label: "Analyze" },
    { href: "/about", label: "About" },
  ]

  return (
    <header className="sticky top-0 z-40 border-b border-border bg-background/85 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <Link href="/" className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Activity className="h-4 w-4" aria-hidden="true" />
          </span>
          <span className="text-base font-semibold tracking-tight">
            RetinaScope<span className="text-primary">-AI</span>
          </span>
          <Badge
            variant="secondary"
            className="hidden sm:inline-flex bg-accent text-accent-foreground border-0 text-[10px] font-semibold uppercase tracking-wider"
          >
            <ShieldCheck className="mr-1 h-3 w-3" aria-hidden="true" />
            Decision Support
          </Badge>
        </Link>

        <nav
          className="flex items-center gap-1 sm:gap-2"
          aria-label="Primary navigation"
        >
          {navItems.map((item) => {
            const active =
              pathname === item.href || pathname.startsWith(item.href + "/")
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                  active
                    ? "text-foreground bg-secondary"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary/60",
                )}
              >
                {item.label}
              </Link>
            )
          })}

          {caseId && (
            <span className="ml-2 hidden md:inline-flex items-center rounded-md border border-border bg-secondary/40 px-2 py-1 font-mono text-[11px] text-muted-foreground">
              Case {caseId}
            </span>
          )}

          {showExport && (
            <Button
              variant="outline"
              size="sm"
              className="ml-2"
              onClick={onExport}
            >
              <FileDown className="h-4 w-4" aria-hidden="true" />
              Export PDF
            </Button>
          )}
        </nav>
      </div>
    </header>
  )
}
