# RetinaScope-AI — Agent Task File
## Tasks 1 & 2: Vessel Mask Visualization Fix + LLM Integration

**Project root:**
```
C:\Users\slima\Downloads\WEB_APP retino\retinascope-ai\
```

**Before starting:** Read every file you are about to modify first. Do not overwrite logic that already works — only add or replace what is explicitly described below.

---

## TASK 1 — Fix Vessel & Lesion Mask Visual Appearance

### Problem
The vessel mask currently renders as a flat grayscale blob on the frontend. It has no color, no overlay on the original fundus image, and gives no clinical information at a glance. Lesion channels have the same problem — they appear as undifferentiated white patches.

### What the agent must do

#### Step 1 — Read the existing visualizer
Open and fully read:
```
backend/app/core/visualizer.py
```
Understand what it currently produces (what images/panels it returns, what format, what variable names).

#### Step 2 — Rewrite the visualizer with colored overlays

Replace or extend the visualizer so it produces the following **5 output images**, each as a `numpy array (H, W, 3)` in RGB format:

---

**Output A — `vessel_colored`**
A green overlay of the vessel mask on the original fundus image.

Logic:
```python
def make_vessel_overlay(fundus_rgb: np.ndarray, vessel_mask: np.ndarray) -> np.ndarray:
    """
    fundus_rgb : np.ndarray  shape (H, W, 3)  dtype uint8  values 0-255
    vessel_mask: np.ndarray  shape (H, W)     dtype float  values 0.0-1.0
                 OR shape (H, W) dtype uint8 values 0-255
    Returns    : np.ndarray  shape (H, W, 3)  dtype uint8  RGB overlay
    """
    # Normalize mask to 0-1 float if needed
    if vessel_mask.max() > 1.0:
        mask = vessel_mask.astype(np.float32) / 255.0
    else:
        mask = vessel_mask.astype(np.float32)

    overlay = fundus_rgb.copy().astype(np.float32)

    # Where vessel is predicted: boost green channel, suppress R and B
    vessel_pixels = mask > 0.5
    overlay[vessel_pixels, 0] = overlay[vessel_pixels, 0] * 0.25          # R dim
    overlay[vessel_pixels, 1] = np.clip(
        overlay[vessel_pixels, 1] * 0.5 + 140, 0, 255)                    # G bright
    overlay[vessel_pixels, 2] = overlay[vessel_pixels, 2] * 0.25          # B dim

    # Alpha blend: 55% overlay, 45% original
    result = cv2.addWeighted(
        fundus_rgb.astype(np.float32), 0.45,
        overlay, 0.55, 0
    )
    return np.clip(result, 0, 255).astype(np.uint8)
```

---

**Output B — `vessel_mask_clean`**
The vessel mask alone, rendered as cyan-on-black (not white-on-black). Used for the standalone mask panel.

Logic:
```python
def make_vessel_clean(vessel_mask: np.ndarray) -> np.ndarray:
    if vessel_mask.max() > 1.0:
        m = (vessel_mask.astype(np.float32) / 255.0 * 255).astype(np.uint8)
    else:
        m = (vessel_mask * 255).astype(np.uint8)

    colored = np.zeros((*m.shape, 3), dtype=np.uint8)
    colored[..., 0] = 0    # R
    colored[..., 1] = m    # G  → cyan = G + B
    colored[..., 2] = m    # B
    return colored
```

---

**Output C — `lesion_overlay`**
A single RGB image showing all lesion channels color-coded on top of the fundus image.

Use these exact colors per channel (RGB tuples):
```python
LESION_COLORS = {
    "MA": (255, 50,  50),   # Red         — Microaneurysms
    "HE": (255, 165,  0),   # Orange      — Hemorrhages
    "EX": (255, 255,  0),   # Yellow      — Hard Exudates
    "OD": (50,  255, 128),  # Green       — Optic Disc
}
```

Logic:
```python
def make_lesion_overlay(
    fundus_rgb: np.ndarray,
    lesion_masks: dict,        # {"MA": np.ndarray, "HE": ..., "EX": ..., "OD": ...}
    alpha: float = 0.55
) -> np.ndarray:
    """
    lesion_masks values: shape (H, W), float 0-1 or uint8 0-255
    """
    overlay = fundus_rgb.copy().astype(np.float32)

    for channel_name, color_rgb in LESION_COLORS.items():
        if channel_name not in lesion_masks:
            continue
        m = lesion_masks[channel_name]
        if m.max() > 1.0:
            m = m.astype(np.float32) / 255.0
        pixels = m > 0.5

        for c_idx, c_val in enumerate(color_rgb):
            overlay[pixels, c_idx] = np.clip(
                overlay[pixels, c_idx] * (1 - alpha) + c_val * alpha, 0, 255
            )

    return np.clip(overlay, 0, 255).astype(np.uint8)
```

---

**Output D — `lesion_grid`**
A 2×2 grid image showing each lesion channel individually with its label. Each cell is 256×256. Font the label in white at the top-left of each cell.

Logic:
```python
def make_lesion_grid(
    lesion_masks: dict,     # {"MA": np.ndarray, "HE": ..., "EX": ..., "OD": ...}
    cell_size: int = 256
) -> np.ndarray:
    cells = []
    for name, color in LESION_COLORS.items():
        m = lesion_masks.get(name, np.zeros((512, 512), dtype=np.float32))
        if m.max() > 1.0:
            m = m.astype(np.float32) / 255.0
        # Resize to cell size
        m_resized = cv2.resize(m, (cell_size, cell_size))
        # Colorize
        cell = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
        pixels = m_resized > 0.5
        for c_idx, c_val in enumerate(color):
            cell[pixels, c_idx] = c_val
        # Label
        cv2.putText(cell, name, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cells.append(cell)

    # Arrange 2 rows × 2 cols
    row1 = np.hstack(cells[:2])
    row2 = np.hstack(cells[2:])
    return np.vstack([row1, row2])
```

---

**Output E — `four_panel`**
A 2×2 composite: top-left = original fundus, top-right = vessel overlay, bottom-left = lesion overlay, bottom-right = existing Grad-CAM heatmap. Each panel resized to 512×512 before compositing.

Logic:
```python
def make_four_panel(
    fundus_rgb: np.ndarray,
    vessel_overlay: np.ndarray,
    lesion_overlay: np.ndarray,
    gradcam: np.ndarray           # shape (H, W, 3), already colored
) -> np.ndarray:
    size = 512
    def r(img): return cv2.resize(img, (size, size))
    row1 = np.hstack([r(fundus_rgb), r(vessel_overlay)])
    row2 = np.hstack([r(lesion_overlay), r(gradcam)])
    return np.vstack([row1, row2])
```

---

#### Step 3 — Wire the outputs into the API response

Open:
```
backend/app/main.py   (or wherever the /analyze endpoint lives)
```

The `/analyze` endpoint must now return all 5 images as base64-encoded PNG strings, alongside the existing grade/confidence fields.

Add this helper at the top of the file (or in a utils module):
```python
import base64, io
from PIL import Image

def ndarray_to_b64(arr: np.ndarray) -> str:
    """Convert RGB numpy array to base64 PNG string."""
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
```

The response JSON must include:
```json
{
  "grade": 2,
  "grade_name": "Moderate DR",
  "confidence": 0.91,
  "vessel_dice": 0.8587,
  "lesion_scores": {"MA": 0.423, "HE": 0.524, "EX": 0.534, "OD": 0.949},
  "images": {
    "original":       "<base64 PNG>",
    "vessel_overlay": "<base64 PNG>",
    "vessel_clean":   "<base64 PNG>",
    "lesion_overlay": "<base64 PNG>",
    "lesion_grid":    "<base64 PNG>",
    "four_panel":     "<base64 PNG>",
    "gradcam":        "<base64 PNG>"
  }
}
```

#### Step 4 — Update the frontend image display

Open:
```
frontend/src/app/analyze/page.tsx   (or whichever component renders the results)
```

Find where the vessel mask and lesion mask images are rendered. Replace every `<img>` or `<Image>` tag that currently shows a mask with the new base64 fields.

Map them as follows:
- Vessel panel → use `images.vessel_overlay`
- Vessel standalone → use `images.vessel_clean`
- Lesion panel → use `images.lesion_overlay`
- Lesion grid → use `images.lesion_grid`
- Main composite panel → use `images.four_panel`

Each should render as:
```tsx
<img
  src={`data:image/png;base64,${result.images.vessel_overlay}`}
  alt="Vessel segmentation overlay"
  className="w-full rounded-lg"
/>
```

#### Step 5 — Add a color legend component to the frontend

Below the lesion overlay image, add a small legend showing which color maps to which lesion. Create a new component:

```
frontend/src/components/LesionLegend.tsx
```

```tsx
const LEGEND = [
  { label: "Microaneurysms (MA)", color: "rgb(255,50,50)" },
  { label: "Hemorrhages (HE)",    color: "rgb(255,165,0)" },
  { label: "Hard Exudates (EX)",  color: "rgb(255,255,0)" },
  { label: "Optic Disc (OD)",     color: "rgb(50,255,128)" },
];

export function LesionLegend() {
  return (
    <div className="flex flex-wrap gap-3 mt-2">
      {LEGEND.map(({ label, color }) => (
        <div key={label} className="flex items-center gap-1.5">
          <div
            className="w-3 h-3 rounded-sm flex-shrink-0"
            style={{ backgroundColor: color }}
          />
          <span className="text-xs text-gray-300">{label}</span>
        </div>
      ))}
    </div>
  );
}
```

Import and place `<LesionLegend />` directly below the lesion overlay `<img>` tag.

#### Step 6 — Verify

Run the backend:
```powershell
cd "C:\Users\slima\Downloads\WEB_APP retino\retinascope-ai\backend"
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

Upload a test fundus image to `http://localhost:3000/analyze`. Confirm:
- [ ] Vessel panel shows green vessels overlaid on the fundus (not a gray blob)
- [ ] Lesion panel shows colored patches (red MA, orange HE, yellow EX, green OD)
- [ ] The 2×2 lesion grid shows 4 individually labeled channels
- [ ] The 4-panel composite renders cleanly
- [ ] The color legend appears below the lesion panel

---

## TASK 2 — Wire the LLM Integration (Gemini)

### Problem
The LLM modules are marked as built in the project plan but may not be properly connected to the `/analyze` endpoint, or may silently fail if the Gemini API call errors out without a fallback. This task makes both LLM calls robust and fully wired.

### Context
- The `.env` file at `C:\Users\slima\Downloads\WEB_APP retino\retinascope-ai\.env` holds `GEMINI_API_KEY`
- The existing LLM folder is at `backend/app/llm/`
- The app uses **Gemini Flash-Lite** (`gemini-1.5-flash-8b`) for text, **Gemini Flash** (`gemini-1.5-flash`) for vision

### What the agent must do

#### Step 1 — Read all existing LLM files

Read every file inside:
```
backend/app/llm/
```
Understand what is already implemented, what functions exist, and what is missing or broken.

#### Step 2 — Verify the .env loading

Open `backend/app/core/config.py` or wherever settings are loaded. Confirm that `GEMINI_API_KEY` is read from the environment and passed to `google.generativeai.configure()`. If it is not, add:

```python
import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
```

This must be called **once at startup**, not inside every function call.

#### Step 3 — Implement / fix `clinical_explainer.py`

The file must be at:
```
backend/app/llm/clinical_explainer.py
```

Full implementation — write this exactly, preserving the async signature:

```python
import google.generativeai as genai

DR_GRADE_NAMES = {
    0: "No Diabetic Retinopathy",
    1: "Mild Non-Proliferative DR",
    2: "Moderate Non-Proliferative DR",
    3: "Severe Non-Proliferative DR",
    4: "Proliferative DR",
}

CLINICAL_PROMPT_TEMPLATE = """You are a clinical ophthalmology assistant writing a report for a general practitioner.

Based on the following automated retinal analysis results, write a structured clinical summary in exactly 3 paragraphs:

Paragraph 1 — Diagnosis: State the DR grade ({grade}/4), its clinical name, and what it means for the patient.
Paragraph 2 — Key Findings: Describe the detected lesions and their clinical significance. Use the lesion scores to determine severity (score > 0.5 = significant, 0.3-0.5 = mild, < 0.3 = minimal/absent).
Paragraph 3 — Recommendation: Suggest appropriate next steps (e.g., ophthalmologist referral urgency, follow-up interval, lifestyle notes).

Analysis Results:
- DR Grade: {grade}/4 — {grade_name}
- Model Confidence: {confidence:.1%}
- Vessel Segmentation Quality (Dice): {vessel_dice:.3f}
- Lesion Detection Scores:
    • Microaneurysms (MA): {ma_score:.3f}
    • Hemorrhages (HE):    {he_score:.3f}
    • Hard Exudates (EX):  {ex_score:.3f}
    • Optic Disc (OD):     {od_score:.3f}

Write in plain English. Be specific and clinically accurate. Do not use bullet points inside the paragraphs. Do not add a title or heading. Start directly with Paragraph 1."""


async def generate_clinical_narrative(
    grade: int,
    confidence: float,
    vessel_dice: float,
    lesion_scores: dict,
) -> str:
    """
    Returns a 3-paragraph clinical narrative string.
    Falls back to a safe default message if the API call fails.
    """
    try:
        prompt = CLINICAL_PROMPT_TEMPLATE.format(
            grade=grade,
            grade_name=DR_GRADE_NAMES.get(grade, "Unknown"),
            confidence=confidence,
            vessel_dice=vessel_dice,
            ma_score=lesion_scores.get("MA", 0.0),
            he_score=lesion_scores.get("HE", 0.0),
            ex_score=lesion_scores.get("EX", 0.0),
            od_score=lesion_scores.get("OD", 0.0),
        )
        model = genai.GenerativeModel("gemini-1.5-flash-8b")
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        # Safe fallback — never let an LLM failure crash the whole analysis
        grade_name = DR_GRADE_NAMES.get(grade, "Unknown")
        return (
            f"Automated analysis detected {grade_name} (Grade {grade}/4) "
            f"with {confidence:.1%} confidence. "
            f"Vessel segmentation quality score: {vessel_dice:.3f}. "
            f"Clinical narrative generation was unavailable at this time. "
            f"Please consult a qualified ophthalmologist for interpretation."
        )
```

#### Step 4 — Implement / fix `visual_analyst.py`

The file must be at:
```
backend/app/llm/visual_analyst.py
```

This sends the `four_panel` image (base64 PNG) to Gemini Vision and returns a short visual description:

```python
import base64
import google.generativeai as genai

DR_GRADE_NAMES = {
    0: "No DR", 1: "Mild DR", 2: "Moderate DR",
    3: "Severe DR", 4: "Proliferative DR",
}

VISUAL_PROMPT_TEMPLATE = """This is a 4-panel retinal fundus analysis image.

Panel layout:
- Top-left:     Original fundus photograph
- Top-right:    Vessel segmentation overlay (green = detected blood vessels)
- Bottom-left:  Lesion map (red = microaneurysms, orange = hemorrhages, yellow = exudates, green = optic disc)
- Bottom-right: Grad-CAM attention heatmap (warm colors = regions that influenced the AI decision)

The AI system graded this retina as: {grade_name} (Grade {grade}/4).

In 3-4 sentences, describe:
1. What anatomical structures are visible in the vessel panel
2. What lesions or abnormalities are visible in the lesion panel
3. Which retinal regions the model focused on according to the heatmap

Keep the description factual, under 80 words, and do not repeat the grade."""


async def analyze_visual_panel(
    four_panel_b64: str,
    grade: int,
) -> str:
    """
    four_panel_b64: base64-encoded PNG string of the 4-panel composite image.
    Returns a short visual description string.
    Falls back gracefully if the API call fails.
    """
    try:
        image_bytes = base64.b64decode(four_panel_b64)
        image_part = {"mime_type": "image/png", "data": image_bytes}

        prompt = VISUAL_PROMPT_TEMPLATE.format(
            grade=grade,
            grade_name=DR_GRADE_NAMES.get(grade, "Unknown"),
        )

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([prompt, image_part])
        return response.text.strip()

    except Exception as e:
        return (
            "Visual analysis of the 4-panel composite image was unavailable at this time. "
            "The panels show vessel segmentation, lesion detection, and model attention regions "
            "as produced by the automated pipeline."
        )
```

#### Step 5 — Wire both LLM calls into the `/analyze` endpoint

Open the `/analyze` endpoint file (likely `backend/app/main.py` or `backend/app/routers/analyze.py`).

After the model inference is complete and you have `grade`, `confidence`, `vessel_dice`, `lesion_scores`, and `four_panel` (base64), add these two async calls:

```python
from app.llm.clinical_explainer import generate_clinical_narrative
from app.llm.visual_analyst import analyze_visual_panel

# Both calls run after inference — wrap in try/except at the endpoint level too
clinical_text = await generate_clinical_narrative(
    grade=grade,
    confidence=confidence,
    vessel_dice=vessel_dice,
    lesion_scores=lesion_scores,
)

visual_text = await analyze_visual_panel(
    four_panel_b64=images["four_panel"],
    grade=grade,
)
```

Add both to the response JSON:
```json
{
  "grade": 2,
  "grade_name": "Moderate Non-Proliferative DR",
  "confidence": 0.91,
  "vessel_dice": 0.8587,
  "lesion_scores": {"MA": 0.423, "HE": 0.524, "EX": 0.534, "OD": 0.949},
  "images": { ... },
  "llm": {
    "clinical_narrative": "<3-paragraph clinical text>",
    "visual_analysis":    "<3-4 sentence visual description>"
  }
}
```

#### Step 6 — Display the LLM outputs on the frontend

Open the analyze results component. Find where the LLM panels are rendered (they may already have placeholder boxes).

**Clinical Narrative panel:**
```tsx
{result.llm.clinical_narrative && (
  <div className="bg-gray-800 rounded-xl p-5 mt-4">
    <h3 className="text-sm font-semibold text-blue-400 uppercase tracking-wide mb-3">
      Clinical Summary
    </h3>
    <p className="text-sm text-gray-200 leading-relaxed whitespace-pre-line">
      {result.llm.clinical_narrative}
    </p>
  </div>
)}
```

**Visual Analysis panel:**
```tsx
{result.llm.visual_analysis && (
  <div className="bg-gray-800 rounded-xl p-5 mt-3">
    <h3 className="text-sm font-semibold text-purple-400 uppercase tracking-wide mb-3">
      Visual Analysis
    </h3>
    <p className="text-sm text-gray-300 leading-relaxed">
      {result.llm.visual_analysis}
    </p>
  </div>
)}
```

Place the Clinical Narrative panel below the grade card. Place the Visual Analysis panel below the 4-panel image.

#### Step 7 — Add a loading state for LLM calls

The Gemini calls may take 3–6 seconds. While they load, show a skeleton/spinner so the UI doesn't feel frozen.

In the frontend, add a boolean `llmLoading` state that is `true` from the moment the `/analyze` request is sent until the full response (including LLM fields) is received. During this time, render:

```tsx
{llmLoading && (
  <div className="bg-gray-800 rounded-xl p-5 mt-4 animate-pulse">
    <div className="h-3 bg-gray-600 rounded w-1/3 mb-3" />
    <div className="h-3 bg-gray-700 rounded w-full mb-2" />
    <div className="h-3 bg-gray-700 rounded w-5/6 mb-2" />
    <div className="h-3 bg-gray-700 rounded w-4/6" />
  </div>
)}
```

#### Step 8 — Verify

Run both backend and frontend. Upload a test fundus image and confirm:
- [ ] After ~4–8 seconds total, the Clinical Summary panel appears with 3 paragraphs of text
- [ ] The Visual Analysis panel appears below the 4-panel image with a 3-4 sentence description
- [ ] If you temporarily set an invalid `GEMINI_API_KEY` in `.env`, both panels show their fallback messages instead of crashing
- [ ] The loading skeleton shows while the LLM calls are in progress
- [ ] The full response JSON at `http://127.0.0.1:8000/api/v1/health` or in the browser network tab includes the `llm` object with both fields populated

---

## Summary Checklist

### Task 1 — Vessel & Lesion Mask Visualization
- [ ] `visualizer.py` — `make_vessel_overlay()` implemented
- [ ] `visualizer.py` — `make_vessel_clean()` implemented
- [ ] `visualizer.py` — `make_lesion_overlay()` with correct colors implemented
- [ ] `visualizer.py` — `make_lesion_grid()` 2×2 grid implemented
- [ ] `visualizer.py` — `make_four_panel()` composite implemented
- [ ] `/analyze` endpoint returns all 7 image fields as base64 PNG
- [ ] Frontend renders `vessel_overlay` (not raw grayscale mask)
- [ ] Frontend renders `lesion_overlay` with colors
- [ ] Frontend renders `lesion_grid`
- [ ] `LesionLegend` component added below lesion panel

### Task 2 — LLM Integration
- [ ] `GEMINI_API_KEY` is loaded from `.env` and passed to `genai.configure()` at startup
- [ ] `clinical_explainer.py` fully implemented with fallback
- [ ] `visual_analyst.py` fully implemented with fallback
- [ ] Both LLM functions called inside the `/analyze` endpoint after inference
- [ ] Response JSON includes `llm.clinical_narrative` and `llm.visual_analysis`
- [ ] Frontend renders Clinical Summary panel (blue header)
- [ ] Frontend renders Visual Analysis panel (purple header)
- [ ] Loading skeleton shown while LLM calls are in progress
- [ ] Fallback messages display correctly when API key is invalid
