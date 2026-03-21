# 🔬 AI-Based Retinal Vessel Segmentation & Diabetic Retinopathy Detection



> Early detection of Diabetic Retinopathy from fundus images using 
> a two-stage deep learning pipeline.

---

## 🎯 Project Overview

This project builds a two-stage AI system that:
1. **Stage 1** — Segments retinal blood vessels and detects lesions
2. **Stage 2** — Grades Diabetic Retinopathy severity (Grade 0–4)

**Target metrics:**
- Stage 1: Dice Score > 0.82
- Stage 2: AUC-ROC > 0.95

---

## 👥 Team

| Role | Responsibility |
|------|----------------|
| Software Engineer | Model development, training, deployment |
| Electrical Engineer | Hardware integration, edge deployment |

---

## 🗂️ Dataset

### Stage 1 — Segmentation
| Source | Images | Mask Type |
|--------|--------|-----------|
| RETINOMIX-5 | 233 | Vessel masks |
| Retinal Vessel Segmentation Combined | 73 | Vessel masks |
| IDRiD | 81 | Lesion masks (5 types) |
| MAPLES-DR | 198 | Vessel + Lesion masks (7 types) |
| **Total** | **585** | — |

📦 **Download:** https://kaggle.com/aymenslimani/dr-stage1-unified-v2

---

## 🏗️ Architecture
```
Fundus Image
     │
     ▼
┌─────────────────┐
│    Stage 1      │  U-Net + TransUNet Ensemble
│  Segmentation   │  → Vessel Map + Lesion Masks
└─────────────────┘
     │
     ▼
┌─────────────────┐
│    Stage 2      │  EfficientNet-B4 + ConvNeXt Ensemble
│   DR Grading    │  → Grade 0 / 1 / 2 / 3 / 4
└─────────────────┘
     │
     ▼
  Grad-CAM Heatmap
```

---

## 🚀 Project Roadmap

- [x] Phase 1 — Data collection and unification (585 images, 7 lesion types)
- [ ] Phase 2 — Preprocessing (CLAHE, ROI masking, augmentation)
- [ ] Phase 3 — Stage 1 model training (vessel + lesion segmentation)
- [ ] Phase 4 — Stage 2 model training (DR grading)
- [ ] Phase 5 — Pipeline integration
- [ ] Phase 6 — Evaluation and validation
- [ ] Phase 7 — Deployment (Web app + Edge device)

---

## ⚙️ Setup
```bash
git clone https://github.com/aymenslimani/retinal-dr-detection
cd retinal-dr-detection
pip install -r requirements.txt
```

---

## 📁 Project Structure
```
retinal-dr-detection/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb      ← coming soon
│   ├── 03_stage1_training.ipynb    ← coming soon
│   └── 04_stage2_training.ipynb    ← coming soon
├── src/
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── configs/
│   ├── stage1_config.yaml
│   └── stage2_config.yaml
└── data/
    └── README.md
```

---

## 📄 License
MIT License
"""

# Save to your project folder
with open(r'C:\path\to\retinal-dr-detection\README.md', 'w', encoding='utf-8') as f:
    f.write(readme)

print("✅ README.md updated!")


