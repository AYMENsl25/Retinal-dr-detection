"use client"

import { useRef, useState } from "react"
import { MessageSquare, Send, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import type { AnalyzeResponse, ChatMessage } from "@/lib/types"
import { cn } from "@/lib/utils"

const SUGGESTIONS = [
  "When should this patient be seen again?",
  "How confident is the model in this grade?",
  "What treatment options apply here?",
  "Explain the vascular biomarkers.",
]

export function ConsultationChat({ caseContext }: { caseContext: AnalyzeResponse }) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [streaming, setStreaming] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  async function send(text: string) {
    const trimmed = text.trim()
    if (!trimmed || streaming) return

    const userMsg: ChatMessage = { role: "user", content: trimmed }
    const next: ChatMessage[] = [...messages, userMsg, { role: "assistant", content: "" }]
    setMessages(next)
    setInput("")
    setStreaming(true)

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [...messages, userMsg],
          caseContext,
        }),
      })
      if (!res.body) throw new Error("No stream")
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let acc = ""
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        acc += decoder.decode(value, { stream: true })
        setMessages((curr) => {
          const updated = [...curr]
          updated[updated.length - 1] = { role: "assistant", content: acc }
          return updated
        })
        requestAnimationFrame(() => {
          scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })
        })
      }
    } catch {
      setMessages((curr) => {
        const updated = [...curr]
        updated[updated.length - 1] = {
          role: "assistant",
          content: "Sorry — the consultation service is unavailable right now.",
        }
        return updated
      })
    } finally {
      setStreaming(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <MessageSquare className="h-4 w-4 text-primary" aria-hidden="true" />
          Consultation Chat
        </CardTitle>
        <CardDescription>
          Ask follow-up questions scoped to this case. Streamed via /api/chat.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div
          ref={scrollRef}
          className="h-72 overflow-y-auto rounded-md border border-border bg-muted/20 p-3 space-y-3"
          aria-live="polite"
        >
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center gap-3 text-center">
              <span className="flex h-9 w-9 items-center justify-center rounded-full bg-primary/10 text-primary">
                <Sparkles className="h-4 w-4" aria-hidden="true" />
              </span>
              <p className="text-sm font-medium">Ask anything about this case</p>
              <div className="flex flex-wrap items-center justify-center gap-1.5 max-w-md">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => send(s)}
                    disabled={streaming}
                    className="rounded-full border border-border bg-card px-2.5 py-1 text-[11px] text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-50"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((m, i) => (
              <div
                key={i}
                className={cn(
                  "flex",
                  m.role === "user" ? "justify-end" : "justify-start",
                )}
              >
                <div
                  className={cn(
                    "max-w-[85%] rounded-lg px-3 py-2 text-sm leading-relaxed",
                    m.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-card border border-border text-foreground",
                  )}
                >
                  {m.content || (streaming && i === messages.length - 1 ? <TypingDots /> : null)}
                </div>
              </div>
            ))
          )}
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault()
            send(input)
          }}
          className="flex items-end gap-2"
        >
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                send(input)
              }
            }}
            placeholder="Ask about follow-up, treatment, or biomarkers…"
            rows={1}
            className="min-h-[40px] resize-none"
            disabled={streaming}
            aria-label="Chat message"
          />
          <Button type="submit" disabled={!input.trim() || streaming} size="icon" aria-label="Send">
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}

function TypingDots() {
  return (
    <span className="inline-flex items-center gap-1" aria-label="Assistant is typing">
      <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground/60 animate-bounce [animation-delay:-0.2s]" />
      <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground/60 animate-bounce [animation-delay:-0.1s]" />
      <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground/60 animate-bounce" />
    </span>
  )
}
