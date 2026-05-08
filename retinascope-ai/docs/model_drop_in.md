# Model Drop-In Checklist

1. Copy checkpoints into `backend/checkpoints/`.
2. Rename them to:
   - `vessel_unet.pth`
   - `lesion_unet.pth`
   - `grader_cnn.pth`
3. Create `backend/checkpoints/preprocess.json`.
4. Paste architecture classes into:
   - `backend/app/models/vessel_unet.py`
   - `backend/app/models/lesion_unet.py`
   - `backend/app/models/grader_cnn.py`
5. Update `backend/app/models/registry.py` to instantiate each model and map checkpoint keys.
6. Run:

```powershell
cd retinascope-ai\backend
python ..\scripts\validate_checkpoints.py
```

