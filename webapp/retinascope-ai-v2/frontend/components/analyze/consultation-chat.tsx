"use client"

import { useState, useRef, useEffect } from "react"
import { useChat } from "@ai-sdk/react"
import { DefaultChatTransport, type UIMessage } from "ai"
import { MessageSquare, Send, User, Sparkles } from "lucide-react"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import type {
  AnalysisResult,
  ClinicalReportData,
  VascularReportData,
} from "@/lib/types"
import { GRADE_NAMES } from "@/lib/types"

interface ConsultationChatProps {
  result: AnalysisResult
  clinical: ClinicalReportData | null
  vascular: VascularReportData | null
}

const SUGGESTED = [
  "When should this patient be seen again?",
  "How confident is the model in this grade?",
  "Explain the vascular biomarkers.",
  "What treatment options apply here?",
]

export function ConsultationChat({
  result,
  clinical,
  vascular,
}: ConsultationChatProps) {
  const [input, setInput] = useState("")
  const scrollerRef = useRef<HTMLDivElement>(null)

  const context = {
    case_id: result.case_id,
    grade: result.grade,
    grade_name: GRADE_NAMES[result.grade],
    confidence: result.calibrated_confidence,
    closeness: result.closeness_to_next_grade,
    decision_flag: result.decision_flag,
    biomarkers: result.biomarkers,
    clinical_summary: clinical?.summary,
    vascular_rationale: vascular?.rationale,
    damage_score: vascular?.overall_damage_score,
  }

  const { messages, sendMessage, status, error } = useChat({
    transport: new DefaultChatTransport({
      api: "/api/chat",
      prepareSendMessagesRequest: ({ messages: ms }) => ({
        body: { messages: ms, context },
      }),
    }),
  })

  const streaming = status === "streaming" || status === "submitted"

  useEffect(() => {
    const el = scrollerRef.current
    if (!el) return
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" })
  }, [messages])

  const send = (text: string) => {
    if (!text.trim() || streaming) return
    sendMessage({ text })
    setInput("")
  }

  return (
    <Card className="border-border">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-md bg-accent text-accent-foreground">
              <MessageSquare className="h-4 w-4" aria-hidden="true" />
            </span>
            <div>
              <div className="text-sm font-semibold">Consultation Chat</div>
              <div className="text-[11px] text-muted-foreground">
                Streaming follow-up Q&amp;A · scoped to case {result.case_id}
              </div>
            </div>
          </div>
          <Badge variant="secondary" className="hidden sm:inline-flex text-[10px]">
            <Sparkles className="mr-1 h-3 w-3" aria-hidden="true" />
            Case-scoped
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        <div
          ref={scrollerRef}
          className="h-[280px] overflow-y-auto rounded-md border border-border bg-secondary/30"
        >
          <div className="space-y-4 p-4">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center py-6 text-center">
                <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-accent">
                  <Sparkles
                    className="h-4 w-4 text-accent-foreground"
                    aria-hidden="true"
                  />
                </div>
                <div className="text-sm font-medium">
                  Ask anything about this case
                </div>
                <div className="mt-1 text-xs text-muted-foreground">
                  Replies stream live. Suggested prompts:
                </div>
                <div className="mt-4 flex flex-wrap justify-center gap-2">
                  {SUGGESTED.map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => send(s)}
                      className="rounded-full border border-border bg-card px-3 py-1 text-xs font-medium text-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m) => (
              <MessageBubble key={m.id} message={m} />
            ))}

            {error && (
              <div className="rounded-md border border-destructive/30 bg-destructive/5 p-3 text-xs text-destructive">
                {error.message || "Chat failed. Please try again."}
              </div>
            )}
          </div>
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault()
            send(input)
          }}
          className="flex items-center gap-2"
        >
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about follow-up, treatment, or biomarkers…"
            aria-label="Ask a question about this case"
            disabled={streaming}
          />
          <Button
            type="submit"
            disabled={streaming || !input.trim()}
            aria-label="Send message"
          >
            <Send className="h-4 w-4" aria-hidden="true" />
            <span className="sr-only sm:not-sr-only sm:ml-1">Send</span>
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}

function MessageBubble({ message }: { message: UIMessage }) {
  const text =
    message.parts
      ?.filter((p): p is { type: "text"; text: string } => p.type === "text")
      .map((p) => p.text)
      .join("") || ""

  const isUser = message.role === "user"

  return (
    <div
      className={cn(
        "flex gap-2.5",
        isUser ? "flex-row-reverse" : "flex-row",
      )}
    >
      <span
        className={cn(
          "flex h-7 w-7 shrink-0 items-center justify-center rounded-full",
          isUser
            ? "bg-foreground text-background"
            : "bg-primary text-primary-foreground",
        )}
        aria-hidden="true"
      >
        {isUser ? (
          <User className="h-3.5 w-3.5" />
        ) : (
          <Sparkles className="h-3.5 w-3.5" />
        )}
      </span>
      <div
        className={cn(
          "max-w-[85%] rounded-lg px-3 py-2 text-sm leading-relaxed",
          isUser
            ? "bg-foreground text-background"
            : "bg-card border border-border text-foreground",
        )}
      >
        <div className="mb-0.5 text-[10px] font-bold uppercase tracking-wider opacity-60">
          {isUser ? "You" : "RetinaScope-AI"}
        </div>
        {text ? (
          <p className="whitespace-pre-wrap">{text}</p>
        ) : (
          <span
            className="inline-block h-3 w-1.5 animate-pulse bg-primary"
            aria-label="Streaming response"
          />
        )}
      </div>
    </div>
  )
}
