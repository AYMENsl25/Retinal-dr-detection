# Vessel Damage Detection

The most reliable v2 design is a hybrid detector:

1. Use your vessel U-Net to create a clean binary vessel mask.
2. Skeletonize the mask.
3. Build a graph from skeleton pixels.
4. Detect candidates from measurable signals:
   - disconnected components
   - excessive endpoints
   - abrupt local vessel disappearance
   - abnormal segment tortuosity
   - caliber change from distance transform width maps
   - quadrant-level density asymmetry
5. Send candidates and biomarkers to an LLM for report wording and JSON cleanup.
6. Draw red ellipses only after coordinate validation.

The starter code is:

```text
backend/app/core/vessel_analysis.py
backend/app/core/metrics.py
backend/app/core/visualizer.py
backend/app/llm/vascular_analyst.py
```

## Current Model Config Impact

Your `pipeline_preprocess.json` sets the vessel model to `TransUNet` with a 3-channel RGB input and no CLAHE. That is compatible with the v2 vessel-first webapp.

The grading model is different: `ConvNeXtBase9Channel` expects 9 channels:

```text
RGB + vessel + MA + HE + EX + OD + CW
```

That means vessel damage detection can run with the vessel model alone, but the final DR grading model needs either lesion masks or a deliberate zero-mask fallback for the five lesion channels.

## Next Improvement

Once you have annotated examples of damaged vessel regions, train a small detector:

```text
YOLO segmentation or RT-DETR for damage bounding boxes
```

Use the deterministic skeleton/graph detector as candidate generation and the learned detector as confirmation.
