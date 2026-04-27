# Brain Tumor Radiogenomics App

Multi-task MRI-based glioma biomarker prediction web app.

Predicts:
- IDH mutation
- MGMT methylation
- Tumor Grade

Built with:
- PyTorch
- Streamlit
- NIfTI preprocessing
- Ensemble inference
- GradCAM explainability


# radiogenomics_app

Streamlit deployment app for brain tumor radiogenomics, aligned to your paper + notebook final pipeline.

## Final Pipeline Used

Priority resolution applied:
1. Paper (`docs/ml_paper8.pdf`) as source of truth
2. Notebook (`notebooks/radiogenomics (4).ipynb`) for implementation details
3. Local deployment checkpoints (`weights/`)

Implemented final configuration:
- Backbone: **ResNet-50 v2 multi-task**
- Input: **4-channel MRI** `(T1, T1ce, T2, FLAIR)` with shape `(B, 4, 224, 224)`
- Heads: `IDH`, `MGMT`, `Grade`
- Ensemble: **2-seed models** (`seed42`, `seed2024`)
- TTA: **horizontal flip**
- Final IDH threshold: **0.49**
- GradCAM target layer: **`backbone.layer4[-1]`**
- Preprocessing: seg-guided tumor-center slice, non-zero z-score normalization, resize to `224x224`

## Project Structure

```
radiogenomics_app/
├── app.py
├── model.py
├── preprocess.py
├── inference.py
├── gradcam.py
├── utils.py
├── requirements.txt
├── README.md
├── docs/
├── notebooks/
├── weights/
├── data/
└── reports/
```

## Required Weights

Place in `weights/`:
- `best_resnet50_v2.pth`
- `best_resnet50_seed42.pth`
- `best_resnet50_seed2024.pth`

Supported fallback aliases:
- `resnet50_v2.pth` (single)
- `ensemble1.pth` (seed42 slot)
- `ensemble2.pth` (seed2024 slot)

## Data Input

Option A: Local patient folder under `data/`, containing:
- `*_t1.nii` or `*_t1.nii.gz`
- `*_t1ce.nii` or `*_t1ce.nii.gz`
- `*_t2.nii` or `*_t2.nii.gz`
- `*_flair.nii` or `*_flair.nii.gz`
- optional: `*_seg.nii` or `*_seg.nii.gz`

Option B: Upload those files directly in Streamlit.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Outputs

- IDH / MGMT / Grade predictions + probabilities
- Ensemble GradCAM visualization
- Exportable PDF report saved in `reports/`
