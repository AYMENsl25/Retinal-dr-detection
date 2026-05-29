"use client"

import { useRef, useState } from "react"
import { UploadCloud, FileImage, ShieldCheck } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import { toast } from "sonner"

interface UploadDropzoneProps {
  onSelect: (dataUrl: string, file: File) => void
}

const ACCEPTED = ["image/png", "image/jpeg", "image/tiff", "image/webp"]
const MAX_BYTES = 15 * 1024 * 1024

export function UploadDropzone({ onSelect }: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  const handleFile = (file?: File | null) => {
    if (!file) return
    if (!ACCEPTED.includes(file.type) && !file.type.startsWith("image/")) {
      toast.error("Unsupported file type", {
        description: "Please upload a PNG, JPG, TIFF or WebP fundus image.",
      })
      return
    }
    if (file.size > MAX_BYTES) {
      toast.error("File too large", {
        description: "Maximum upload size is 15 MB.",
      })
      return
    }
    const reader = new FileReader()
    reader.onload = (e) => {
      const result = e.target?.result
      if (typeof result === "string") onSelect(result, file)
    }
    reader.onerror = () => toast.error("Could not read file")
    reader.readAsDataURL(file)
  }

  return (
    <Card className="border-border">
      <CardContent className="p-6 sm:p-8">
        <div className="mb-5">
          <div className="text-sm font-semibold">1 · Upload fundus image</div>
          <p className="mt-1 text-sm text-muted-foreground">
            Color fundus photography, single eye. The pipeline will preprocess,
            segment vessels, grade ICDR severity, and produce structured
            clinical reasoning.
          </p>
        </div>

        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragging(false)
            handleFile(e.dataTransfer.files?.[0])
          }}
          className={cn(
            "group flex w-full flex-col items-center justify-center rounded-xl border-2 border-dashed p-10 text-center transition-colors",
            dragging
              ? "border-primary bg-accent"
              : "border-border bg-secondary/40 hover:border-primary/50 hover:bg-accent/50",
          )}
          aria-label="Upload retinal fundus image"
        >
          <div
            className={cn(
              "mb-4 flex h-14 w-14 items-center justify-center rounded-full border bg-card transition-colors",
              dragging ? "border-primary" : "border-border",
            )}
          >
            <UploadCloud
              className={cn(
                "h-6 w-6 transition-colors",
                dragging ? "text-primary" : "text-muted-foreground",
              )}
              aria-hidden="true"
            />
          </div>
          <div className="text-base font-medium">
            Drop a retinal fundus image, or click to browse
          </div>
          <div className="mt-1.5 text-xs text-muted-foreground">
            PNG · JPG · TIFF · WebP — single eye, ideally 512×512 or larger ·
            max 15 MB
          </div>
          <div className="mt-4 flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <ShieldCheck className="h-3.5 w-3.5 text-primary" aria-hidden="true" />
            Decision support only — not a medical diagnosis.
          </div>
        </button>

        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/tiff,image/webp,image/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />

        <div className="mt-6 flex flex-wrap items-center gap-x-3 gap-y-2 text-[11px] text-muted-foreground">
          <FileImage className="h-3.5 w-3.5" aria-hidden="true" />
          <span className="font-medium">Pipeline:</span>
          {["Preprocess", "Segment", "Grade", "Reason", "Report"].map(
            (s, i, arr) => (
              <span key={s} className="flex items-center gap-2">
                <span className="rounded bg-secondary px-1.5 py-0.5 font-mono text-foreground">
                  {i + 1} · {s}
                </span>
                {i < arr.length - 1 && (
                  <span className="text-border">›</span>
                )}
              </span>
            ),
          )}
        </div>
      </CardContent>
    </Card>
  )
}
