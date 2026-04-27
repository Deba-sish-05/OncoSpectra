# Brain Tumor Radiogenomics App

Streamlit deployment for multi-task glioma biomarker prediction from MRI:
- IDH mutation
- MGMT methylation
- Tumor Grade

## Final Pipeline (Preserved)

Priority used for implementation:
1. `docs/ml_paper8.pdf` (final source of truth)
2. `notebooks/radiogenomics (4).ipynb` (code reference)
3. `weights/` checkpoints (deployment)

Backend preserved:
- ResNet-50 v2 multi-task model
- 4-channel MRI input: `T1, T1ce, T2, FLAIR`
- 2-seed ensemble inference (`seed42`, `seed2024`)
- Horizontal flip TTA
- IDH threshold `0.49`
- Existing preprocessing and prediction logic unchanged

## Project Root Structure

```
RADIOGENOMICS APP/
├── app.py
├── model.py
├── preprocess.py
├── inference.py
├── gradcam.py
├── utils.py
├── requirements.txt
├── README.md
├── assets/
├── data/
├── docs/
├── notebooks/
├── reports/
└── weights/
```

## Data Input

Use a case folder inside `data/` with:
- `*_t1.nii` or `*_t1.nii.gz`
- `*_t1ce.nii` or `*_t1ce.nii.gz`
- `*_t2.nii` or `*_t2.nii.gz`
- `*_flair.nii` or `*_flair.nii.gz`
- optional `*_seg.nii` or `*_seg.nii.gz`

Or upload those files directly in the app.

## Required Weights

Place checkpoints in `weights/`:
- `best_resnet50_v2.pth`
- `best_resnet50_seed42.pth`
- `best_resnet50_seed2024.pth`

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Outputs

- Prediction cards with confidence
- Localized GradCAM++ visualization
- PDF report saved under `reports/`
