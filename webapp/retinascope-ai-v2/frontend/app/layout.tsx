import type { Metadata, Viewport } from "next"
import { Analytics } from "@vercel/analytics/next"
import { Toaster } from "@/components/ui/sonner"
import "./globals.css"

export const metadata: Metadata = {
  title: "RetinaScope-AI — Diabetic Retinopathy Decision Support",
  description:
    "Clinical decision-support web app for diabetic retinopathy screening: vessel segmentation, ICDR grading, calibrated confidence, and dual-LLM clinical reasoning. Decision support only — not a medical diagnosis.",
  applicationName: "RetinaScope-AI",
  keywords: [
    "diabetic retinopathy",
    "DR screening",
    "ICDR",
    "vessel segmentation",
    "ophthalmology",
    "clinical decision support",
    "AI medical imaging",
  ],
  authors: [{ name: "RetinaScope-AI" }],
  robots: { index: true, follow: true },
}

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#0B6E6E" },
    { media: "(prefers-color-scheme: dark)", color: "#0F8F8F" },
  ],
  width: "device-width",
  initialScale: 1,
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html
      lang="en"
      className="bg-background"
      suppressHydrationWarning
    >
      <body className="font-sans antialiased min-h-screen bg-background text-foreground">
        {children}
        <Toaster richColors position="top-right" />
        {process.env.NODE_ENV === "production" && <Analytics />}
      </body>
    </html>
  )
}
