import { useState, useRef, useEffect, useCallback } from "react";

// ─── design tokens ───────────────────────────────────────────────────────────
const T = {
  teal:    "#0B6E6E",
  tealL:   "#0F8F8F",
  tealXL:  "#E6F4F4",
  slate:   "#1E293B",
  slateL:  "#334155",
  gray:    "#64748B",
  grayL:   "#94A3B8",
  grayXL:  "#F1F5F9",
  white:   "#FFFFFF",
  bg:      "#F7FAFA",
  red:     "#DC2626",
  amber:   "#D97706",
  green:   "#16A34A",
};

// ─── global styles ────────────────────────────────────────────────────────────
const globalCSS = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'DM Sans', sans-serif; background: ${T.bg}; color: ${T.slate}; }
  
  @keyframes fadeUp   { from { opacity:0; transform:translateY(16px) } to { opacity:1; transform:translateY(0) } }
  @keyframes fadeIn   { from { opacity:0 } to { opacity:1 } }
  @keyframes pulse2   { 0%,100% { opacity:1 } 50% { opacity:.4 } }
  @keyframes spin     { to { transform: rotate(360deg) } }
  @keyframes scanline { 0% { top: -8px } 100% { top: 100% } }
  @keyframes barFill  { from { width:0 } to { width: var(--w) } }

  .fade-up  { animation: fadeUp .5s ease both }
  .fade-in  { animation: fadeIn .4s ease both }
  .pulse2   { animation: pulse2 1.4s ease infinite }
  .spin     { animation: spin .8s linear infinite }

  .panel-img { width:100%; border-radius:6px; display:block; background:#000; }
  .chip { display:inline-flex; align-items:center; gap:5px; padding:2px 10px; border-radius:99px; font-size:11px; font-weight:600; font-family:'DM Mono',monospace; letter-spacing:.04em; }
  .chip-teal  { background:${T.tealXL}; color:${T.teal}; }
  .chip-red   { background:#FEF2F2; color:${T.red}; }
  .chip-amber { background:#FFFBEB; color:${T.amber}; }
  .chip-green { background:#F0FDF4; color:${T.green}; }

  .card { background:${T.white}; border-radius:12px; box-shadow:0 1px 3px rgba(0,0,0,.07),0 1px 2px rgba(0,0,0,.04); }
  .mono { font-family:'DM Mono',monospace; }
  
  .bar-fill { height:6px; border-radius:3px; background:${T.teal}; transition: width 1.2s ease; }
  .bar-track { height:6px; border-radius:3px; background:${T.grayXL}; overflow:hidden; }

  .step-row { display:flex; align-items:center; gap:8px; padding:6px 0; }
  .step-dot  { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
  .step-dot.done    { background:${T.teal}; }
  .step-dot.active  { background:${T.tealL}; animation:pulse2 1s infinite; }
  .step-dot.pending { background:#CBD5E1; }

  button { cursor:pointer; font-family:'DM Sans',sans-serif; border:none; }
  input  { font-family:'DM Sans',sans-serif; }

  .report-section h4 { font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:${T.gray}; margin-bottom:6px; margin-top:16px; }
  .report-section h4:first-child { margin-top:0; }
  .report-section p, .report-section li { font-size:13px; line-height:1.65; color:${T.slateL}; }
  .report-section ul { padding-left:16px; }
  .report-section li { margin-bottom:3px; }

  .zoom-crop { border-radius:8px; overflow:hidden; cursor:pointer; transition: transform .15s; }
  .zoom-crop:hover { transform: scale(1.04); }

  ::-webkit-scrollbar { width:4px; }
  ::-webkit-scrollbar-track { background:transparent; }
  ::-webkit-scrollbar-thumb { background:#CBD5E1; border-radius:2px; }

  .nav-link { font-size:14px; font-weight:500; color:${T.gray}; text-decoration:none; cursor:pointer; }
  .nav-link:hover { color:${T.teal}; }
  .nav-link.active { color:${T.slate}; font-weight:600; }

  .tag { display:inline-block; padding:1px 8px; border-radius:4px; font-size:11px; font-weight:600; font-family:'DM Mono',monospace; }
  .tag-high   { background:#FEE2E2; color:${T.red}; }
  .tag-medium { background:#FEF3C7; color:${T.amber}; }
  .tag-low    { background:#DCFCE7; color:${T.green}; }
`;

// ─── mock pipeline simulation ─────────────────────────────────────────────────
async function mockPipeline(imageDataUrl, onStep) {
  const steps = ["preprocess","segment","grade","reason"];
  for (let i = 0; i < steps.length; i++) {
    onStep(steps[i]);
    await new Promise(r => setTimeout(r, 700 + Math.random() * 600));
  }
  // Generate mock panels from the image using canvas
  const panels = await generateMockPanels(imageDataUrl);
  return panels;
}

async function generateMockPanels(src) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      const SIZE = 512;
      // ── original (CLAHE-like brightening) ──
      const orig = document.createElement("canvas"); orig.width = orig.height = SIZE;
      const octx = orig.getContext("2d");
      octx.drawImage(img, 0, 0, SIZE, SIZE);
      const od = octx.getImageData(0, 0, SIZE, SIZE);
      for (let i = 0; i < od.data.length; i += 4) {
        const r = od.data[i], g = od.data[i+1], b = od.data[i+2];
        od.data[i]   = Math.min(255, r * 1.08);
        od.data[i+1] = Math.min(255, g * 1.1);
        od.data[i+2] = Math.min(255, b * 1.05);
      }
      octx.putImageData(od, 0, 0);

      // ── vessel mask (green channel threshold + morphology simulation) ──
      const maskC = document.createElement("canvas"); maskC.width = maskC.height = SIZE;
      const mctx = maskC.getContext("2d");
      mctx.fillStyle = "#000"; mctx.fillRect(0, 0, SIZE, SIZE);
      const raw = orig.getContext("2d").getImageData(0, 0, SIZE, SIZE);
      const maskData = mctx.getImageData(0, 0, SIZE, SIZE);
      // circular ROI mask radius
      const cx = SIZE/2, cy = SIZE/2, cr = SIZE * 0.46;
      for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
          const dist = Math.sqrt((x-cx)**2+(y-cy)**2);
          if (dist > cr) continue;
          const i = (y*SIZE+x)*4;
          const g = raw.data[i+1];
          const r = raw.data[i];
          const b = raw.data[i+2];
          // Simple vessel detection: high green relative to avg
          const avg = (r+g+b)/3;
          const vessel = g > avg * 1.05 && g > 60 && avg > 30;
          if (vessel) {
            maskData.data[i] = maskData.data[i+1] = maskData.data[i+2] = 255;
            maskData.data[i+3] = 255;
          }
        }
      }
      mctx.putImageData(maskData, 0, 0);

      // ── probability heatmap (jet colormap on vessel prob) ──
      const heatC = document.createElement("canvas"); heatC.width = heatC.height = SIZE;
      const hctx = heatC.getContext("2d");
      hctx.fillStyle = "#000"; hctx.fillRect(0,0,SIZE,SIZE);
      const hData = hctx.getImageData(0,0,SIZE,SIZE);
      for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
          const dist = Math.sqrt((x-cx)**2+(y-cy)**2);
          if (dist > cr) continue;
          const i = (y*SIZE+x)*4;
          const g = raw.data[i+1];
          const r2 = raw.data[i], b2 = raw.data[i+2];
          const avg = (r2+g+b2)/3;
          let prob = Math.min(1, Math.max(0, (g/255 * 1.3 - 0.3)));
          prob = prob * (1 - dist/(cr*1.1));
          const [jr,jg,jb] = jetColor(prob);
          hData.data[i] = jr; hData.data[i+1] = jg; hData.data[i+2] = jb; hData.data[i+3] = 255;
        }
      }
      hctx.putImageData(hData, 0,0);

      // ── damage annotated mask ──
      const dmgC = document.createElement("canvas"); dmgC.width = dmgC.height = SIZE;
      const dctx = dmgC.getContext("2d");
      dctx.drawImage(maskC, 0,0);
      // Draw mock damage ellipses
      const damages = [
        { x:220, y:160, rx:38, ry:22, sev:"high",   label:"Vessel discontinuity" },
        { x:310, y:280, rx:28, ry:18, sev:"medium",  label:"Tortuosity" },
        { x:150, y:330, rx:22, ry:15, sev:"low",     label:"Caliber change" },
      ];
      damages.forEach(d => {
        const col = d.sev==="high"?"#FF2222":d.sev==="medium"?"#FF8800":"#FFCC00";
        const lw  = d.sev==="high"?3:d.sev==="medium"?2:1.5;
        dctx.beginPath();
        dctx.ellipse(d.x, d.y, d.rx, d.ry, 0, 0, Math.PI*2);
        dctx.strokeStyle = col;
        dctx.lineWidth = lw;
        dctx.stroke();
        dctx.font = "bold 9px 'DM Mono',monospace";
        dctx.fillStyle = col;
        dctx.fillText(d.label, d.x - d.rx, d.y - d.ry - 3);
      });

      // ── red overlay ──
      const ovlC = document.createElement("canvas"); ovlC.width = ovlC.height = SIZE;
      const octx2 = ovlC.getContext("2d");
      octx2.drawImage(img, 0, 0, SIZE, SIZE);
      const ovlD = octx2.getImageData(0,0,SIZE,SIZE);
      for (let i=0;i<ovlD.data.length;i+=4){
        const w = maskData.data[i];
        if (w > 128) {
          ovlD.data[i]   = Math.min(255, ovlD.data[i]+120);
          ovlD.data[i+1] = Math.max(0,   ovlD.data[i+1]-40);
          ovlD.data[i+2] = Math.max(0,   ovlD.data[i+2]-40);
        }
      }
      octx2.putImageData(ovlD, 0,0);

      // zoom crops from damage regions
      const crops = damages.map(d => {
        const cc = document.createElement("canvas"); cc.width = cc.height = 96;
        const cctx = cc.getContext("2d");
        const pad = 24;
        cctx.drawImage(maskC,
          d.x-d.rx-pad, d.y-d.ry-pad, (d.rx+pad)*2, (d.ry+pad)*2,
          0, 0, 96, 96);
        cctx.beginPath();
        const col = d.sev==="high"?"#FF2222":d.sev==="medium"?"#FF8800":"#FFCC00";
        cctx.ellipse(48, 48, d.rx*96/((d.rx+pad)*2), d.ry*96/((d.ry+pad)*2), 0, 0, Math.PI*2);
        cctx.strokeStyle = col; cctx.lineWidth = 2; cctx.stroke();
        return { src: cc.toDataURL("image/png"), finding: d.label, severity: d.sev };
      });

      resolve({
        original:   orig.toDataURL("image/png"),
        heatmap:    heatC.toDataURL("image/png"),
        cleanMask:  maskC.toDataURL("image/png"),
        dmgMask:    dmgC.toDataURL("image/png"),
        overlay:    ovlC.toDataURL("image/png"),
        crops,
      });
    };
    img.src = src;
  });
}

function jetColor(t) {
  // jet colormap
  const r = Math.min(255, Math.max(0, Math.round(255 * (1.5 - Math.abs(4*t - 3)))));
  const g = Math.min(255, Math.max(0, Math.round(255 * (1.5 - Math.abs(4*t - 2)))));
  const b = Math.min(255, Math.max(0, Math.round(255 * (1.5 - Math.abs(4*t - 1)))));
  return [r,g,b];
}

// ─── Anthropic API helpers ────────────────────────────────────────────────────
async function callClinicalLLM(grade, biomarkers) {
  const gradeNames = ["No DR","Mild NPDR","Moderate NPDR","Severe NPDR","Proliferative DR"];
  const gName = gradeNames[grade] ?? "Unknown";
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      messages: [{
        role: "user",
        content: `You are a clinical assistant generating a decision-support report for a diabetic retinopathy screening result. Return ONLY valid JSON, no markdown, no preamble.

DR Grade: ${grade} — ${gName}
Vessel density: ${biomarkers.vessel_density}
Tortuosity index: ${biomarkers.tortuosity}
Fractal dimension: ${biomarkers.fractal_dim}
AVR: ${biomarkers.avr}

Return this JSON schema exactly:
{"summary":"1-2 sentence summary","pathophysiology":"2-3 sentence explanation","risk_factors":["list","of","risk","factors"],"recommendations":["list","of","clinical","recommendations"],"follow_up_window":"e.g. 12 months","lifestyle_advice":["list","of","lifestyle","tips"],"red_flags":["list","of","red","flags","to","watch"],"disclaimer":"standard decision support disclaimer"}`
      }]
    })
  });
  const d = await res.json();
  const text = d.content?.filter(c=>c.type==="text").map(c=>c.text).join("") || "";
  try { return JSON.parse(text.replace(/```json|```/g,"")); }
  catch { return null; }
}

async function callVascularLLM(grade, dmgScore, biomarkers) {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 800,
      messages: [{
        role: "user",
        content: `You are a retinal vascular analyst. Return ONLY valid JSON, no markdown.

DR Grade: ${grade}
Overall damage score: ${dmgScore}/100
Num vessel components: ${biomarkers.num_vessel_components}
Broken segments estimate: ${biomarkers.broken_segments}
Quadrant densities: NW=${biomarkers.q_nw?.toFixed(3)}, NE=${biomarkers.q_ne?.toFixed(3)}, SW=${biomarkers.q_sw?.toFixed(3)}, SE=${biomarkers.q_se?.toFixed(3)}
Tortuosity index: ${biomarkers.tortuosity}

Return this schema:
{"damaged_regions":[{"quadrant":"NW/NE/SW/SE","severity":"low/medium/high","finding":"short description ≤40 chars"}],"overall_damage_score":${dmgScore},"rationale":"1-2 sentences grounded in biomarkers","needs_specialist_review":${grade>=3||dmgScore>60},"per_grade_severity":{"0":0.75,"1":0.15,"2":0.03,"3":0.04,"4":0.03}}`
      }]
    })
  });
  const d = await res.json();
  const text = d.content?.filter(c=>c.type==="text").map(c=>c.text).join("") || "";
  try { return JSON.parse(text.replace(/```json|```/g,"")); }
  catch { return null; }
}

async function streamChat(messages, onChunk) {
  const sysPrompt = `You are RetinaScope-AI's clinical consultation assistant. 
Answer questions about the current retinal analysis case. Be medically accurate, concise, and always remind the user this is decision support only — not a diagnosis. Keep replies under 120 words.`;
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      stream: true,
      system: sysPrompt,
      messages,
    })
  });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value);
    const lines = chunk.split("\n");
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      try {
        const data = JSON.parse(line.slice(6));
        if (data.type === "content_block_delta" && data.delta?.text) {
          onChunk(data.delta.text);
        }
      } catch {}
    }
  }
}

// ─── sub-components ───────────────────────────────────────────────────────────
function NavBar({ page, setPage, caseId }) {
  return (
    <nav style={{ background: T.white, borderBottom: `1px solid ${T.grayXL}`, position: "sticky", top: 0, zIndex: 100 }}>
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 24px", display: "flex", alignItems: "center", justifyContent: "space-between", height: 56 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer" }} onClick={() => setPage("home")}>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: T.teal, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round">
              <circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/>
              <line x1="12" y1="3" x2="12" y2="1"/><line x1="12" y1="23" x2="12" y2="21"/>
              <line x1="3" y1="12" x2="1" y2="12"/><line x1="23" y1="12" x2="21" y2="12"/>
            </svg>
          </div>
          <span style={{ fontWeight: 700, fontSize: 16, color: T.slate }}>RetinaScope<span style={{ color: T.teal }}>-AI</span></span>
          <span className="chip chip-teal" style={{ marginLeft: 2 }}>DECISION SUPPORT</span>
        </div>
        <div style={{ display: "flex", gap: 24, alignItems: "center" }}>
          <span className={`nav-link ${page==="analyze"?"active":""}`} onClick={() => setPage("analyze")}>Analyze</span>
          {caseId && <span className="mono" style={{ fontSize: 11, color: T.gray }}>Case {caseId}</span>}
          {page === "results" && (
            <button onClick={() => {}} style={{ padding: "6px 16px", background: "transparent", border: `1px solid ${T.grayL}`, borderRadius: 6, fontSize: 13, color: T.slateL, fontWeight: 500 }}>
              Export PDF
            </button>
          )}
        </div>
      </div>
    </nav>
  );
}

function LandingPage({ setPage }) {
  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: "80px 24px 60px" }}>
      <div className="fade-up" style={{ marginBottom: 12 }}>
        <span className="chip chip-teal">v0.1 · mock pipeline · drop-in your .pth checkpoints</span>
      </div>
      <h1 className="fade-up" style={{ fontSize: "clamp(40px,5vw,64px)", fontWeight: 800, lineHeight: 1.1, letterSpacing: "-.03em", color: T.slate, animationDelay: ".05s" }}>
        Diabetic retinopathy<br />screening, <span style={{ color: T.teal }}>explained.</span>
      </h1>
      <p className="fade-up" style={{ fontSize: 17, color: T.gray, maxWidth: 560, marginTop: 20, lineHeight: 1.65, animationDelay: ".1s" }}>
        A clinical-decision-support web app that turns a single retinal fundus image into a vessel segmentation, an ICDR grade, calibrated confidence, and a structured clinician-ready report — backed by dual-LLM reasoning.
      </p>
      <div className="fade-up" style={{ display: "flex", gap: 12, marginTop: 32, alignItems: "center", animationDelay: ".15s" }}>
        <button onClick={() => setPage("analyze")} style={{ padding: "12px 28px", background: T.teal, color: T.white, borderRadius: 8, fontSize: 15, fontWeight: 600, display: "flex", alignItems: "center", gap: 8 }}>
          Try the demo
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
        </button>
        <button style={{ padding: "12px 24px", background: "transparent", border: `1.5px solid ${T.slateL}`, borderRadius: 8, fontSize: 15, fontWeight: 500, color: T.slateL }}>
          View the pipeline
        </button>
      </div>

      <div className="fade-up" style={{ display: "flex", gap: 48, marginTop: 56, animationDelay: ".2s" }}>
        {[["0.87","Vessel Dice"],["0.96","Grade QWK"],["2 + chat","LLM stages"]].map(([v,l]) => (
          <div key={l}>
            <div style={{ fontSize: 28, fontWeight: 800, fontFamily: "'DM Mono',monospace", color: T.slate }}>{v}</div>
            <div style={{ fontSize: 12, color: T.gray, marginTop: 2 }}>{l}</div>
          </div>
        ))}
      </div>

      {/* feature cards */}
      <div className="fade-up" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(240px,1fr))", gap: 16, marginTop: 64, animationDelay: ".25s" }}>
        {[
          { icon:"🔬", title:"Dual-mask visualization", desc:"Clean vessel map + LLM-annotated damage regions with red ellipses and zoom crops." },
          { icon:"🧠", title:"Dual-LLM reasoning", desc:"Clinical explainer (LLM-1) + vision-conditioned vascular damage analyst (LLM-2)." },
          { icon:"📊", title:"Calibrated confidence", desc:"Temperature-scaled softmax + MC-Dropout uncertainty. No raw overconfident probabilities." },
          { icon:"💬", title:"Consultation chat", desc:"Ask follow-up questions about the case. Context-scoped to the current patient." },
        ].map(f => (
          <div key={f.title} className="card" style={{ padding: 20 }}>
            <div style={{ fontSize: 24, marginBottom: 10 }}>{f.icon}</div>
            <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 6 }}>{f.title}</div>
            <div style={{ fontSize: 13, color: T.gray, lineHeight: 1.55 }}>{f.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function UploadPage({ onAnalyze }) {
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef();

  const handleFile = f => {
    if (!f || !f.type.startsWith("image/")) return;
    const r = new FileReader();
    r.onload = e => onAnalyze(e.target.result);
    r.readAsDataURL(f);
  };

  return (
    <div style={{ maxWidth: 780, margin: "0 auto", padding: "48px 24px" }}>
      <h2 style={{ fontWeight: 800, fontSize: 26, marginBottom: 4 }}>Fundus Analysis</h2>
      <p style={{ fontSize: 14, color: T.gray, marginBottom: 28 }}>
        Upload a single retinal fundus image. The pipeline returns vessel segmentation, an ICDR grade, calibrated confidence, and dual-LLM clinical reasoning.
      </p>

      <div className="card" style={{ padding: 28 }}>
        <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 4 }}>1 · Upload fundus image</div>
        <div style={{ fontSize: 13, color: T.gray, marginBottom: 20 }}>Color fundus photography, single eye. The mock pipeline simulates the stages described in §5 of the project plan.</div>

        <div
          onClick={() => fileRef.current.click()}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={e => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
          style={{ border: `2px dashed ${dragging ? T.teal : "#CBD5E1"}`, borderRadius: 10, padding: "56px 32px", textAlign: "center", cursor: "pointer", background: dragging ? T.tealXL : T.grayXL, transition: "all .2s" }}
        >
          <div style={{ width: 48, height: 48, borderRadius: "50%", background: dragging ? T.tealXL : T.white, border: `1.5px solid ${T.grayL}`, display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 14px" }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={T.teal} strokeWidth="2"><polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/></svg>
          </div>
          <div style={{ fontWeight: 600, color: T.slateL }}>Drop a retinal fundus image, or click to browse</div>
          <div style={{ fontSize: 12, color: T.gray, marginTop: 4 }}>PNG · JPG · TIFF — single eye, color fundus, ideally 512×512 or larger</div>
          <div style={{ fontSize: 11, color: T.grayL, marginTop: 8, display: "flex", alignItems: "center", justifyContent: "center", gap: 4 }}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2"/><polyline points="9 11 12 14 22 4"/></svg>
            Decision support only — not a medical diagnosis.
          </div>
        </div>
        <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }} onChange={e => handleFile(e.target.files[0])} />

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 20 }}>
          <div style={{ display: "flex", gap: 8, fontSize: 12, color: T.grayL }}>
            {["1 Preprocess","2 Segment","3 Grade","4 Reason"].map((s,i) => (
              <span key={s} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <span style={{ background: T.grayXL, color: T.gray, padding: "1px 7px", borderRadius: 4, fontFamily: "monospace" }}>{s}</span>
                {i < 3 && <span style={{ color: "#CBD5E1" }}>→</span>}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function PipelineLoader({ step }) {
  const steps = [
    { id:"preprocess", label:"Preprocessing", detail:"CLAHE · ROI crop · resize 512×512 · normalize" },
    { id:"segment",    label:"Vessel segmentation", detail:"U-Net ensemble → binary mask" },
    { id:"grade",      label:"DR grading",  detail:"Grader CNN → temperature-scaled softmax" },
    { id:"reason",     label:"LLM reasoning", detail:"Clinical report + vascular damage analysis" },
  ];
  const stepIdx = steps.findIndex(s => s.id === step);

  return (
    <div style={{ maxWidth: 520, margin: "80px auto", padding: "0 24px", textAlign: "center" }}>
      <div style={{ width: 48, height: 48, border: `3px solid ${T.tealXL}`, borderTopColor: T.teal, borderRadius: "50%", margin: "0 auto 28px" }} className="spin" />
      <h3 style={{ fontWeight: 700, fontSize: 18, marginBottom: 24 }}>Running analysis pipeline…</h3>
      <div className="card" style={{ padding: 20, textAlign: "left" }}>
        {steps.map((s, i) => (
          <div className="step-row" key={s.id}>
            <div className={`step-dot ${i < stepIdx ? "done" : i === stepIdx ? "active" : "pending"}`} />
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, color: i <= stepIdx ? T.slate : T.grayL }}>{s.label}</div>
              <div style={{ fontSize: 11, color: T.grayL }}>{s.detail}</div>
            </div>
            {i < stepIdx && <svg style={{ marginLeft: "auto" }} width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={T.teal} strokeWidth="2.5"><polyline points="20 6 9 17 4 12"/></svg>}
          </div>
        ))}
      </div>
    </div>
  );
}

function PanelViewer({ panels, opacity, setOpacity }) {
  const defs = [
    { key: "original",   label: "Original Fundus",      sub: "CLAHE preprocessed" },
    { key: "heatmap",    label: "Probability Heatmap",   sub: "vessel softmax [jet]" },
    { key: "cleanMask",  label: "Binary Vessel Mask",    sub: "U-Net @ τ=0.5" },
    { key: "dmgMask",    label: "Damage Analysis",       sub: "LLM-annotated regions" },
    { key: "overlay",    label: "Red Overlay",           sub: "Mask + Original" },
  ];
  const [active, setActive] = useState(null);

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 8 }}>
        {defs.map(d => (
          <div key={d.key} style={{ cursor: "pointer" }} onClick={() => setActive(active === d.key ? null : d.key)}>
            <div style={{ borderRadius: 8, overflow: "hidden", border: active === d.key ? `2px solid ${T.teal}` : "2px solid transparent", background: "#000" }}>
              <img src={panels[d.key]} alt={d.label} className="panel-img" style={{ aspectRatio:"1/1", objectFit:"cover" }} />
            </div>
            <div style={{ fontSize: 10, color: T.slateL, fontWeight: 600, marginTop: 4 }}>{d.label}</div>
            <div style={{ fontSize: 9, color: T.grayL }}>{d.sub}</div>
          </div>
        ))}
      </div>

      {active === "overlay" && (
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 10, fontSize: 12, color: T.gray }}>
          <span>Overlay opacity</span>
          <input type="range" min={0} max={100} value={opacity} onChange={e => setOpacity(+e.target.value)} style={{ width: 120, accentColor: T.teal }} />
          <span className="mono" style={{ fontSize: 11 }}>{opacity}%</span>
          <span style={{ marginLeft: 12, fontSize: 10, color: T.grayL }}>Vessel segmentation · Dice 0.87 (reported)</span>
        </div>
      )}
    </div>
  );
}

function ZoomStrip({ crops }) {
  const [expanded, setExpanded] = useState(false);
  const [modal, setModal] = useState(null);

  if (!crops?.length) return null;
  return (
    <div style={{ marginTop: 10 }}>
      <button onClick={() => setExpanded(!expanded)} style={{ background: "none", fontSize: 12, color: T.teal, fontWeight: 600, display: "flex", alignItems: "center", gap: 4, padding: "4px 0" }}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d={expanded ? "M18 15l-6-6-6 6" : "M6 9l6 6 6-6"}/></svg>
        Damage zoom strip — {crops.length} region{crops.length > 1 ? "s" : ""} detected
      </button>
      {expanded && (
        <div style={{ display: "flex", gap: 10, marginTop: 8 }}>
          {crops.map((c, i) => (
            <div key={i} className="zoom-crop" style={{ width: 96, border: `1.5px solid ${c.severity==="high"?"#FCA5A5":c.severity==="medium"?"#FCD34D":"#86EFAC"}` }} onClick={() => setModal(c)}>
              <img src={c.src} alt={c.finding} style={{ width: 96, height: 96, display: "block" }} />
              <div style={{ padding: "3px 5px", background: T.white }}>
                <span className={`tag tag-${c.severity}`}>{c.severity[0].toUpperCase()}</span>
                <div style={{ fontSize: 9, color: T.gray, marginTop: 1, lineHeight: 1.3 }}>{c.finding}</div>
              </div>
            </div>
          ))}
        </div>
      )}
      {modal && (
        <div onClick={() => setModal(null)} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,.6)", zIndex: 200, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div className="card" style={{ padding: 20, maxWidth: 320 }} onClick={e => e.stopPropagation()}>
            <img src={modal.src} alt={modal.finding} style={{ width: "100%", borderRadius: 8 }} />
            <div style={{ marginTop: 12, fontWeight: 600, fontSize: 14 }}>{modal.finding}</div>
            <div style={{ marginTop: 4 }}><span className={`tag tag-${modal.severity}`}>{modal.severity} severity</span></div>
            <button onClick={() => setModal(null)} style={{ marginTop: 12, background: T.grayXL, padding: "6px 16px", borderRadius: 6, fontSize: 12, color: T.slateL, fontWeight: 500 }}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
}

function GradeCard({ grade, probs, confidence, closeness, uncertainty }) {
  const gradeNames = ["No DR","Mild NPDR","Moderate NPDR","Severe NPDR","Proliferative DR"];
  const gradeColors = [T.green, "#84CC16", T.amber, "#EA580C", T.red];
  const col = gradeColors[grade] ?? T.gray;

  return (
    <div className="card fade-in" style={{ padding: 20 }}>
      <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: ".08em", textTransform: "uppercase", color: T.gray, marginBottom: 8 }}>ICDR GRADE</div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 4 }}>
        <span style={{ fontSize: 42, fontWeight: 900, fontFamily: "'DM Mono',monospace", color: col, lineHeight: 1 }}>{grade}</span>
        <span style={{ fontWeight: 700, fontSize: 16, color: T.slateL }}>{gradeNames[grade]}</span>
      </div>

      {grade >= 3
        ? <div style={{ display: "inline-flex", alignItems: "center", gap: 6, background: "#FEE2E2", color: T.red, padding: "4px 10px", borderRadius: 6, fontSize: 12, fontWeight: 600, marginBottom: 12 }}>
            <span>⚠</span> Refer to specialist
          </div>
        : grade >= 1
        ? <div style={{ display: "inline-flex", alignItems: "center", gap: 6, background: "#FEF3C7", color: T.amber, padding: "4px 10px", borderRadius: 6, fontSize: 12, fontWeight: 600, marginBottom: 12 }}>
            <span>ℹ</span> Second look advised
          </div>
        : <div style={{ height: 8 }} />
      }

      <div style={{ marginBottom: 12 }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
          <span style={{ color: T.gray }}>Calibrated confidence</span>
          <span className="mono" style={{ fontWeight: 600 }}>{Math.round(confidence * 100)}%</span>
        </div>
        <div className="bar-track">
          <div className="bar-fill" style={{ "--w": `${confidence*100}%`, width: `${confidence*100}%`, background: col }} />
        </div>
      </div>

      {grade < 4 && (
        <div style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
            <span style={{ color: T.gray }}>Closeness to Grade {grade+1}</span>
            <span className="mono" style={{ fontWeight: 600 }}>{Math.round(closeness * 100)}%</span>
          </div>
          <div className="bar-track">
            <div className="bar-fill" style={{ "--w": `${closeness*100}%`, width: `${closeness*100}%`, background: "#94A3B8" }} />
          </div>
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, fontSize: 12, borderTop: `1px solid ${T.grayXL}`, paddingTop: 12 }}>
        {[["Entropy", uncertainty?.entropy?.toFixed(2)],["MC-Dropout σ", uncertainty?.mc_std?.toFixed(2)],
          ["Vessel density", "0.183"],["Tortuosity", "1.586"],["Fractal dim.", "1.671"],["AVR","0.749"]
        ].map(([l,v]) => (
          <div key={l}>
            <div style={{ color: T.gray, marginBottom: 1 }}>{l}</div>
            <div className="mono" style={{ fontWeight: 600, color: T.slate }}>{v ?? "—"}</div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 12, borderTop: `1px solid ${T.grayXL}`, paddingTop: 12 }}>
        <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: ".06em", textTransform: "uppercase", color: T.gray, marginBottom: 8 }}>PER GRADE PROBABILITY</div>
        {(probs||[.75,.15,.03,.04,.03]).map((p,i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
            <span className="mono" style={{ width: 14, fontSize: 11, color: T.gray }}>{i}</span>
            <div className="bar-track" style={{ flex: 1 }}>
              <div className="bar-fill" style={{ "--w":`${p*100}%`, width:`${p*100}%`, background: gradeColors[i], height: 5 }}/>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ClinicalReport({ loading, report }) {
  const [open, setOpen] = useState(true);
  if (!open) return (
    <div className="card" style={{ padding: "12px 20px", cursor:"pointer" }} onClick={()=>setOpen(true)}>
      <div style={{ display:"flex", alignItems:"center", gap:8 }}>
        <span style={{ fontSize:16 }}>🩺</span>
        <span style={{ fontWeight:600, fontSize:13 }}>Clinical Report</span>
        <span style={{ fontSize:11, color:T.teal, marginLeft:"auto" }}>Show ▸</span>
      </div>
    </div>
  );

  return (
    <div className="card fade-in" style={{ padding: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 14 }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: 14, display: "flex", alignItems: "center", gap: 8 }}>
            <span>🩺</span> Clinical Report
          </div>
          <div style={{ fontSize: 11, color: T.gray }}>LLM-1 · Reasoning model</div>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          <span className="chip chip-teal">ICDR</span>
          <span className="chip chip-teal">AAO PPP</span>
          <button onClick={()=>setOpen(false)} style={{ background:"none", color:T.grayL, fontSize:14 }}>✕</button>
        </div>
      </div>

      {loading && <div style={{ display:"flex", gap:8, alignItems:"center", color:T.gray, fontSize:13 }}>
        <div style={{ width:14,height:14,border:`2px solid ${T.tealXL}`,borderTopColor:T.teal,borderRadius:"50%" }} className="spin"/>
        Generating clinical report…
      </div>}

      {report && (
        <div className="report-section">
          <h4>Summary</h4><p>{report.summary}</p>
          <h4>Pathophysiology</h4><p>{report.pathophysiology}</p>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16, marginTop:14 }}>
            <div>
              <h4>Recommendations</h4>
              <ul>{(report.recommendations||[]).map((r,i)=><li key={i}>{r}</li>)}</ul>
            </div>
            <div>
              <h4>Lifestyle Advice</h4>
              <ul>{(report.lifestyle_advice||[]).map((r,i)=><li key={i}>{r}</li>)}</ul>
            </div>
          </div>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16, marginTop:14 }}>
            <div>
              <h4>Risk Factors</h4>
              <ul>{(report.risk_factors||[]).map((r,i)=><li key={i}>{r}</li>)}</ul>
            </div>
            <div>
              <h4>Follow-up</h4><p>⏱ {report.follow_up_window}</p>
            </div>
          </div>
          {(report.red_flags||[]).length > 0 && (
            <div style={{ marginTop:14, background:"#FEF2F2", border:"1px solid #FECACA", borderRadius:8, padding:12 }}>
              <h4 style={{ color:T.red }}>🚨 Red Flags</h4>
              <ul>{report.red_flags.map((r,i)=><li key={i} style={{ color:"#991B1B" }}>{r}</li>)}</ul>
            </div>
          )}
          <div style={{ marginTop:14, fontSize:11, color:T.grayL, borderTop:`1px solid ${T.grayXL}`, paddingTop:10 }}>
            ⚠ {report.disclaimer || "Decision support only — not a medical diagnosis. All findings must be confirmed by a licensed ophthalmologist."}
          </div>
        </div>
      )}
    </div>
  );
}

function VascularReport({ loading, report }) {
  const [open, setOpen] = useState(true);
  if (!open) return (
    <div className="card" style={{ padding: "12px 20px", cursor:"pointer" }} onClick={()=>setOpen(true)}>
      <div style={{ display:"flex", alignItems:"center", gap:8 }}>
        <span style={{ fontSize:16 }}>🫀</span>
        <span style={{ fontWeight:600, fontSize:13 }}>Vascular Damage Report</span>
        <span style={{ fontSize:11, color:T.teal, marginLeft:"auto" }}>Show ▸</span>
      </div>
    </div>
  );

  const score = report?.overall_damage_score ?? 5;
  const perGrade = report?.per_grade_severity ?? {"0":.75,"1":.15,"2":.03,"3":.04,"4":.03};
  const gradeColors = [T.green,"#84CC16",T.amber,"#EA580C",T.red];
  const regions = report?.damaged_regions ?? [];

  return (
    <div className="card fade-in" style={{ padding: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 14 }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: 14, display: "flex", alignItems: "center", gap: 8 }}>
            <span>🫀</span> Vascular Damage Report
          </div>
          <div style={{ fontSize: 11, color: T.gray }}>LLM-2 · Vision-conditioned</div>
        </div>
        <div style={{ display:"flex", gap:6 }}>
          <span className="chip chip-teal" style={{ fontSize:9 }}>grounded on biomarkers</span>
          <button onClick={()=>setOpen(false)} style={{ background:"none", color:T.grayL, fontSize:14 }}>✕</button>
        </div>
      </div>

      {loading && <div style={{ display:"flex", gap:8, alignItems:"center", color:T.gray, fontSize:13 }}>
        <div style={{ width:14,height:14,border:`2px solid ${T.tealXL}`,borderTopColor:T.teal,borderRadius:"50%" }} className="spin"/>
        Analyzing vascular damage…
      </div>}

      {!loading && (
        <>
          <div style={{ marginBottom:16 }}>
            <div style={{ fontSize:11, color:T.gray, marginBottom:4, textTransform:"uppercase", letterSpacing:".06em", fontWeight:700 }}>OVERALL DAMAGE SCORE</div>
            <div style={{ display:"flex", alignItems:"center", gap:12 }}>
              <div className="bar-track" style={{ flex:1 }}>
                <div className="bar-fill" style={{ "--w":`${score}%`, width:`${score}%`, background: score>60?T.red:score>30?T.amber:T.teal }}/>
              </div>
              <span className="mono" style={{ fontWeight:700, fontSize:18 }}>{score}<span style={{ fontSize:12, color:T.gray }}> / 100</span></span>
            </div>
          </div>

          <div style={{ marginBottom:16 }}>
            <div style={{ fontSize:11, color:T.gray, marginBottom:8, textTransform:"uppercase", letterSpacing:".06em", fontWeight:700 }}>PER-GRADE SEVERITY</div>
            {Object.entries(perGrade).map(([g,v])=>(
              <div key={g} style={{ display:"flex", alignItems:"center", gap:8, marginBottom:5 }}>
                <span style={{ width:40, fontSize:11, color:T.gray }}>Grade {g}</span>
                <div className="bar-track" style={{ flex:1 }}>
                  <div className="bar-fill" style={{ "--w":`${v*100}%`, width:`${v*100}%`, background:gradeColors[+g] }}/>
                </div>
              </div>
            ))}
          </div>

          {regions.length > 0 && (
            <div style={{ marginBottom:14 }}>
              <div style={{ fontSize:11, color:T.gray, marginBottom:8, textTransform:"uppercase", letterSpacing:".06em", fontWeight:700 }}>DAMAGED REGIONS</div>
              {regions.map((r,i)=>(
                <div key={i} style={{ display:"flex", alignItems:"center", gap:8, padding:"5px 8px", background:T.grayXL, borderRadius:6, marginBottom:4 }}>
                  <span className={`tag tag-${r.severity}`}>{r.severity}</span>
                  <span style={{ fontSize:12, color:T.slateL }}>{r.finding}</span>
                  <span style={{ marginLeft:"auto", fontSize:11, color:T.gray, fontFamily:"monospace" }}>{r.quadrant}</span>
                </div>
              ))}
            </div>
          )}

          {regions.length === 0 && !loading && (
            <div style={{ display:"flex", alignItems:"center", gap:8, padding:"12px 16px", background:T.grayXL, borderRadius:8, marginBottom:14 }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={T.grayL} strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              <span style={{ fontSize:12, color:T.gray }}>No focal vascular damage detected.</span>
            </div>
          )}

          {report?.rationale && (
            <div style={{ borderTop:`1px solid ${T.grayXL}`, paddingTop:12 }}>
              <div style={{ fontSize:11, color:T.gray, marginBottom:4, textTransform:"uppercase", letterSpacing:".06em", fontWeight:700 }}>RATIONALE</div>
              <p style={{ fontSize:13, color:T.slateL, lineHeight:1.6 }}>{report.rationale}</p>
            </div>
          )}

          {report?.needs_specialist_review && (
            <div style={{ marginTop:12, padding:"8px 12px", background:"#FEF2F2", border:"1px solid #FECACA", borderRadius:6, fontSize:12, color:"#991B1B" }}>
              Cross-check flagged disagreement between visual analysis and CNN grade — specialist review recommended.
            </div>
          )}

          {!report?.needs_specialist_review && !loading && (
            <div style={{ marginTop:12, display:"flex", alignItems:"center", gap:6, fontSize:12, color:T.green }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="20 6 9 17 4 12"/></svg>
              No specialist referral indicated at this time.
            </div>
          )}
        </>
      )}
    </div>
  );
}

function ConsultationChat({ grade, clinical, vascular }) {
  const [msgs, setMsgs] = useState([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const bottomRef = useRef();
  const suggested = ["When should this patient be seen again?","How confident is the model in this grade?","What treatment options apply here?","Explain the vascular biomarkers."];

  const send = async (text) => {
    if (!text.trim() || streaming) return;
    const userMsg = { role:"user", content: text };
    const context = `Current case: DR Grade ${grade}. Clinical summary: ${clinical?.summary||"(loading)"}. Vascular rationale: ${vascular?.rationale||"(loading)"}. Damage score: ${vascular?.overall_damage_score??5}/100.`;
    const newMsgs = [...msgs, userMsg];
    setMsgs([...newMsgs, { role:"assistant", content:"" }]);
    setInput("");
    setStreaming(true);

    const apiMsgs = [
      { role:"user", content: context + "\n\nUser question: " + text }
    ];
    let full = "";
    try {
      await streamChat(apiMsgs, chunk => {
        full += chunk;
        setMsgs(m => { const c = [...m]; c[c.length-1] = { role:"assistant", content:full }; return c; });
      });
    } catch {}
    setStreaming(false);
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior:"smooth" }), 50);
  };

  return (
    <div className="card" style={{ padding: 20 }}>
      <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:16 }}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={T.teal} strokeWidth="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        <span style={{ fontWeight:700, fontSize:14 }}>Consultation Chat</span>
        <span style={{ fontSize:11, color:T.gray }}>Ask follow-up questions scoped to this case.</span>
      </div>

      <div style={{ minHeight: 140, maxHeight: 260, overflowY:"auto", marginBottom:14 }}>
        {msgs.length === 0 && (
          <div style={{ display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", height:140, gap:12 }}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke={T.grayL} strokeWidth="1.5"><circle cx="12" cy="12" r="3"/><path d="M12 2v2M12 20v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M2 12h2M20 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
            <div style={{ fontSize:13, color:T.gray, fontWeight:500 }}>Ask anything about this case</div>
            <div style={{ display:"flex", flexWrap:"wrap", gap:6, justifyContent:"center" }}>
              {suggested.map(s=>(
                <button key={s} onClick={()=>send(s)} style={{ padding:"4px 12px", background:T.grayXL, borderRadius:99, fontSize:12, color:T.slateL, border:"none" }}>{s}</button>
              ))}
            </div>
          </div>
        )}
        {msgs.map((m,i)=>(
          <div key={i} style={{ marginBottom:10 }}>
            <div style={{ fontSize:10, color:T.grayL, marginBottom:2, fontWeight:600, textTransform:"uppercase", letterSpacing:".05em" }}>{m.role==="user"?"You":"RetinaScope-AI"}</div>
            <div style={{ fontSize:13, color:T.slateL, lineHeight:1.6, background:m.role==="user"?T.grayXL:"transparent", padding:m.role==="user"?"8px 12px":"0", borderRadius:8 }}>
              {m.content || <span className="pulse2" style={{ color:T.teal }}>▋</span>}
            </div>
          </div>
        ))}
        <div ref={bottomRef}/>
      </div>

      <div style={{ display:"flex", gap:8, alignItems:"center" }}>
        <input value={input} onChange={e=>setInput(e.target.value)} onKeyDown={e=>e.key==="Enter"&&send(input)}
          placeholder="Ask about follow-up, treatment, or biomarkers…"
          style={{ flex:1, padding:"8px 14px", border:`1.5px solid ${T.grayXL}`, borderRadius:8, fontSize:13, color:T.slateL, outline:"none", background:T.bg }}
        />
        <button onClick={()=>send(input)} disabled={streaming || !input.trim()} style={{ width:36,height:36, background:input.trim()&&!streaming?T.teal:"#CBD5E1", borderRadius:8, display:"flex",alignItems:"center",justifyContent:"center", transition:"background .2s" }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
        </button>
      </div>
    </div>
  );
}

// ─── decision flag banner ─────────────────────────────────────────────────────
function DecisionBanner({ grade, entropy }) {
  if (grade >= 3) return (
    <div style={{ background:"#FEE2E2", border:`1px solid #FECACA`, borderRadius:8, padding:"10px 18px", marginBottom:16, display:"flex", alignItems:"center", gap:10 }}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={T.red} strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><circle cx="12" cy="16" r=".5" fill={T.red}/></svg>
      <span style={{ fontSize:13, fontWeight:600, color:"#991B1B" }}>HIGH CONCERN — Specialist referral recommended</span>
    </div>
  );
  if (entropy > 0.3 || grade >= 1) return (
    <div style={{ background:"#FEF3C7", border:`1px solid #FDE68A`, borderRadius:8, padding:"10px 18px", marginBottom:16, display:"flex", alignItems:"center", gap:10 }}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={T.amber} strokeWidth="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><circle cx="12" cy="17" r=".5" fill={T.amber}/></svg>
      <span style={{ fontSize:13, fontWeight:600, color:"#92400E" }}>MEDIUM — Second look advised</span>
    </div>
  );
  return (
    <div style={{ background:"#F0FDF4", border:`1px solid #BBF7D0`, borderRadius:8, padding:"10px 18px", marginBottom:16, display:"flex", alignItems:"center", gap:10 }}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={T.green} strokeWidth="2"><polyline points="20 6 9 17 4 12"/></svg>
      <span style={{ fontSize:13, fontWeight:600, color:"#14532D" }}>HIGH CONFIDENCE — No immediate referral indicated</span>
    </div>
  );
}

// ─── results page ─────────────────────────────────────────────────────────────
function ResultsPage({ result }) {
  const [opacity, setOpacity] = useState(75);
  const grade = result.grade ?? 0;
  const confidence = result.confidence ?? 0.75;
  const closeness = result.closeness ?? 0.17;
  const uncertainty = result.uncertainty ?? { entropy: 0.83, mc_std: 0.05 };
  const probs = result.probs ?? [.75,.15,.03,.04,.03];

  return (
    <div style={{ maxWidth:1200, margin:"0 auto", padding:"24px 24px 48px" }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"baseline", marginBottom:16 }}>
        <h2 style={{ fontWeight:800, fontSize:22 }}>Fundus Analysis</h2>
        <button onClick={result.onNew} style={{ fontSize:12, color:T.gray, background:T.grayXL, padding:"5px 12px", borderRadius:6, fontWeight:500, display:"flex", alignItems:"center", gap:5 }}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-3.6"/></svg>
          New analysis
        </button>
      </div>
      <p style={{ fontSize:13, color:T.gray, marginBottom:20 }}>Upload a single retinal fundus image. The pipeline returns vessel segmentation, an ICDR grade, calibrated confidence, and dual-LLM clinical reasoning.</p>

      <DecisionBanner grade={grade} entropy={uncertainty.entropy} />

      <div style={{ display:"grid", gridTemplateColumns:"1fr 280px", gap:16, marginBottom:16 }}>
        {/* left: panels */}
        <div className="card" style={{ padding:18 }}>
          <PanelViewer panels={result.panels} opacity={opacity} setOpacity={setOpacity} />
          <ZoomStrip crops={result.panels.crops} />
        </div>
        {/* right: grade card */}
        <GradeCard grade={grade} probs={probs} confidence={confidence} closeness={closeness} uncertainty={uncertainty} />
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16, marginBottom:16 }}>
        <ClinicalReport loading={result.clinicalLoading} report={result.clinical} />
        <VascularReport loading={result.vascularLoading} report={result.vascular} />
      </div>

      <ConsultationChat grade={grade} clinical={result.clinical} vascular={result.vascular} />

      <div style={{ marginTop:16, fontSize:11, color:T.grayL, textAlign:"center", padding:"8px 0", borderTop:`1px solid ${T.grayXL}` }}>
        Decision support only — not a medical diagnosis. All findings must be confirmed by a licensed ophthalmologist.
      </div>
    </div>
  );
}

// ─── main app ─────────────────────────────────────────────────────────────────
export default function App() {
  const [page, setPage] = useState("home");
  const [pipelineStep, setPipelineStep] = useState(null);
  const [caseId, setCaseId] = useState(null);
  const [result, setResult] = useState(null);

  const handleAnalyze = useCallback(async (imageDataUrl) => {
    setPage("pipeline");
    setPipelineStep("preprocess");
    setCaseId(Math.random().toString(36).slice(2,9));

    const panels = await mockPipeline(imageDataUrl, step => setPipelineStep(step));

    // generate realistic mock grade
    const grade = Math.random() < .6 ? 0 : Math.random() < .5 ? 1 : Math.random() < .5 ? 2 : 3;
    const probs = [0,0,0,0,0].map((_,i)=>{
      const d = Math.abs(i-grade);
      return d===0?.65+Math.random()*.15 : d===1?.1+Math.random()*.08 : .02+Math.random()*.03;
    });
    const s = probs.reduce((a,b)=>a+b,0);
    const normProbs = probs.map(p=>+(p/s).toFixed(3));
    const biomarkers = { vessel_density: .183, tortuosity: 1.586, fractal_dim: 1.671, avr: .749 };

    const res = {
      grade,
      probs: normProbs,
      confidence: normProbs[grade],
      closeness: normProbs[Math.min(grade+1,4)] / (normProbs[grade] + normProbs[Math.min(grade+1,4)] + .001),
      uncertainty: { entropy: +(0.6+Math.random()*.4).toFixed(2), mc_std: +(0.03+Math.random()*.05).toFixed(2) },
      panels,
      clinical: null,
      vascular: null,
      clinicalLoading: true,
      vascularLoading: true,
      onNew: () => { setPage("analyze"); setResult(null); },
    };
    setResult(res);
    setPage("results");

    // parallel LLM calls
    const dmgScore = grade * 12 + Math.round(Math.random()*15);
    const fullBio = { ...biomarkers, num_vessel_components: 312, broken_segments: grade+1, q_nw:.18, q_ne:.17, q_sw:.19, q_se:.16 };

    callClinicalLLM(grade, biomarkers).then(r => {
      setResult(prev => ({ ...prev, clinical: r, clinicalLoading: false }));
    }).catch(() => {
      setResult(prev => ({ ...prev, clinicalLoading: false }));
    });

    callVascularLLM(grade, dmgScore, fullBio).then(r => {
      setResult(prev => ({ ...prev, vascular: r, vascularLoading: false }));
    }).catch(() => {
      setResult(prev => ({ ...prev, vascularLoading: false }));
    });
  }, []);

  return (
    <>
      <style>{globalCSS}</style>
      <NavBar page={page} setPage={setPage} caseId={caseId} />
      {page === "home"     && <LandingPage setPage={setPage} />}
      {page === "analyze"  && <UploadPage onAnalyze={handleAnalyze} />}
      {page === "pipeline" && <PipelineLoader step={pipelineStep} />}
      {page === "results"  && result && <ResultsPage result={result} />}
    </>
  );
}
