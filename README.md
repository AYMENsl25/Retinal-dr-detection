# 🔬 AI-Based Retinal Vessel Segmentation & Diabetic Retinopathy Detection

> Early detection of Diabetic Retinopathy from fundus images using 
> a two-stage deep learning pipeline.

---

## 🎯 Project Overview

This project builds a two-stage AI system that:
1. **Stage 1** — Segments retinal blood vessels and detects lesions 
   (microaneurysms, haemorrhages, exudates, optic disc)
2. **Stage 2** — Grades Diabetic Retinopathy severity (Grade 0–4)

**Target metrics:**
- Stage 1: Dice Score > 0.82
- Stage 2: AUC-ROC > 0.95

---

## 👥 Team

| Role | Responsibility |
|------|----------------|
| Software Engineer | Model development, training, deployment |
| Electrical Engineer | Hardware integration, edge deployment (Jetson Nano) |

---

## 🗂️ Dataset

### Stage 1 — Segmentation
| Source | Images | Mask Type |
|--------|--------|-----------|
| RETINOMIX-5 | 233 | Vessel masks |
| Retinal Vessel Segmentation Combined | 73 | Vessel masks |
| IDRiD (Segmentation) | 81 | Lesion masks (5 types) |
| **Total** | **387** | — |

### Stage 2 — Grading (coming soon)
| Source | Images | Labels |
|--------|--------|--------|
| IDRiD (Disease Grading) | 516 | Grade 0–4 |
| APTOS 2019 | 3,662 | Grade 0–4 |

> ⚠️ Data is not included in this repository.
> Download links and instructions are in `data/README.md`

---

## 🏗️ Architecture
```
Fundus Image
     │
     ▼
┌─────────────┐
│   Stage 1   │  U-Net + TransUNet Ensemble
│ Segmentation│  → Vessel Map + Lesion Masks
└─────────────┘
     │
     ▼
┌─────────────┐
│   Stage 2   │  EfficientNet-B4 + ConvNeXt Ensemble
│  DR Grading │  → Grade 0 / 1 / 2 / 3 / 4
└─────────────┘
     │
     ▼
  Grad-CAM
 Heatmap Output
```

---

## 🚀 Project Roadmap

- [x] Phase 1 — Data collection & unification
- [ ] Phase 2 — Preprocessing (CLAHE, ROI, augmentation)
- [ ] Phase 3 — Stage 1 model training
- [ ] Phase 4 — Stage 2 model training
- [ ] Phase 5 — Pipeline integration
- [ ] Phase 6 — Evaluation
- [ ] Phase 7 — Deployment

---

## ⚙️ Setup
```bash
git clone https://github.com/YOUR_USERNAME/retinal-dr-detection
cd retinal-dr-detection
pip install -r requirements.txt
```

---

## 📦 Requirements

See `requirements.txt`

---

## 📄 License

MIT License — see LICENSE file
```

---

### Step 4 — Create requirements.txt

In your repo click **Add file** → **Create new file** → name it `requirements.txt`:
```
torch>=2.0.0
torchvision>=0.15.0
segmentation-models-pytorch>=0.3.3
albumentations>=1.3.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
timm>=0.9.0
wandb>=0.15.0
fastapi>=0.100.0
uvicorn>=0.23.0
tqdm>=4.65.0
PyYAML>=6.0
