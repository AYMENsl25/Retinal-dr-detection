"use client"

import { useCallback, useRef, useState } from "react"
import { Upload, ImageIcon, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

interface UploadDropzoneProps {
  file: File | null
  onFileChange: (file: File | null) => void
  disabled?: boolean
}

export function UploadDropzone({ file, onFileChange, disabled }: UploadDropzoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return
      const f = files[0]
      if (!f.type.startsWith("image/")) return
      onFileChange(f)
    },
    [onFileChange],
  )

  const previewUrl = file ? URL.createObjectURL(file) : null

  return (
    <div className="space-y-3">
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label="Upload fundus image"
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => {
          if (disabled) return
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault()
            inputRef.current?.click()
          }
        }}
        onDragOver={(e) => {
          e.preventDefault()
          if (!disabled) setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setIsDragging(false)
          if (!disabled) handleFiles(e.dataTransfer.files)
        }}
        className={cn(
          "relative flex flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed p-8 text-center transition-colors",
          "border-border bg-muted/30 hover:bg-muted/50 hover:border-primary/50 cursor-pointer",
          isDragging && "border-primary bg-primary/5",
          disabled && "opacity-60 cursor-not-allowed",
        )}
      >
        {previewUrl ? (
          <div className="flex w-full items-center gap-4">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewUrl || "/placeholder.svg"}
              alt="Fundus preview"
              className="h-16 w-16 rounded-md object-cover ring-1 ring-border"
            />
            <div className="flex-1 text-left min-w-0">
              <p className="text-sm font-medium truncate">{file?.name}</p>
              <p className="text-xs text-muted-foreground">
                {file ? `${(file.size / 1024).toFixed(1)} KB` : ""}
              </p>
            </div>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              aria-label="Remove image"
              onClick={(e) => {
                e.stopPropagation()
                onFileChange(null)
              }}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        ) : (
          <>
            <span className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-primary">
              <Upload className="h-6 w-6" aria-hidden="true" />
            </span>
            <div className="space-y-1">
              <p className="text-sm font-medium">Drop a retinal fundus image, or click to browse</p>
              <p className="text-xs text-muted-foreground">
                PNG · JPG · TIFF — single eye, color fundus, ideally 512×512 or larger
              </p>
            </div>
            <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
              <ImageIcon className="h-3 w-3" />
              <span>Decision support only — not a medical diagnosis.</span>
            </div>
          </>
        )}

        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
          disabled={disabled}
        />
      </div>
    </div>
  )
}
