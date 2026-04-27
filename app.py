from __future__ import annotations

from datetime import datetime
from pathlib import Path
import uuid

import streamlit as st
import torch

from gradcam import compute_ensemble_gradcam, make_multimodal_rgb, overlay_heatmap
from inference import IDH_THRESHOLD, MGMT_THRESHOLD, ensemble_predict, load_deployment_models
from preprocess import MODALITIES, build_input_tensor, discover_case_files, find_patient_dirs
from utils import ensure_dir, export_pdf_report, resolve_device, save_uploaded_file

APP_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = APP_DIR / "weights"
DATA_DIR = APP_DIR / "data"
REPORTS_DIR = ensure_dir(APP_DIR / "reports")
UPLOAD_DIR = ensure_dir(DATA_DIR / "_uploads")


st.set_page_config(page_title="Radiogenomics App", layout="wide")
st.title("Brain Tumor Radiogenomics App")
st.caption(
    "Paper-aligned final pipeline: ResNet50-v2 architecture, 2-seed ensemble + horizontal-flip TTA, "
    "IDH threshold = 0.49, GradCAM on backbone.layer4[-1]."
)


@st.cache_resource(show_spinner=False)
def _cached_models(weights_dir: str, device_name: str):
    return load_deployment_models(weights_dir=weights_dir, device=torch.device(device_name))


device = resolve_device()
st.write(f"**Device:** `{device}`")

weight_files = sorted(p.name for p in WEIGHTS_DIR.glob("*.pth"))
st.write(f"**Available checkpoints:** {', '.join(weight_files) if weight_files else 'None'}")

try:
    models_bundle = _cached_models(str(WEIGHTS_DIR), str(device))
except Exception as exc:  # pragma: no cover - UI path
    st.error(f"Failed to load model weights: {exc}")
    st.stop()

st.write(
    "**Loaded paths:** "
    f"seed42=`{models_bundle['paths']['seed42']}`, "
    f"seed2024=`{models_bundle['paths']['seed2024']}`, "
    f"single=`{models_bundle['paths']['single']}`"
)

mode = st.radio("Input Source", ["Use sample from data/", "Upload NIfTI files"], horizontal=True)

case_id = "unknown_case"
modality_paths: dict[str, Path] = {}
seg_path: Path | None = None

if mode == "Use sample from data/":
    patient_dirs = [p for p in find_patient_dirs(DATA_DIR) if p.name != "_uploads"]
    if not patient_dirs:
        st.warning("No patient folders found in `data/`.")
    else:
        selected = st.selectbox("Select patient folder", options=patient_dirs, format_func=lambda p: p.name)
        case_id = selected.name
        modality_paths, seg_path = discover_case_files(selected)
        st.write("**Detected files:**")
        for mod in MODALITIES:
            st.write(f"- {mod}: `{modality_paths.get(mod, 'MISSING')}`")
        st.write(f"- seg: `{seg_path if seg_path is not None else 'not provided'}`")
else:
    st.write("Upload all 4 modalities (`.nii` or `.nii.gz`). Segmentation mask is optional.")
    uploads = {}
    col1, col2 = st.columns(2)
    with col1:
        uploads["t1"] = st.file_uploader("T1", type=["nii", "gz"], key="up_t1")
        uploads["t2"] = st.file_uploader("T2", type=["nii", "gz"], key="up_t2")
    with col2:
        uploads["t1ce"] = st.file_uploader("T1ce", type=["nii", "gz"], key="up_t1ce")
        uploads["flair"] = st.file_uploader("FLAIR", type=["nii", "gz"], key="up_flair")
    upload_seg = st.file_uploader("Segmentation mask (optional)", type=["nii", "gz"], key="up_seg")

    if all(uploads.get(m) is not None for m in MODALITIES):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        run_dir = ensure_dir(UPLOAD_DIR / run_id)
        case_id = f"upload_{run_id}"
        for mod in MODALITIES:
            modality_paths[mod] = save_uploaded_file(uploads[mod], run_dir, f"{case_id}_{mod}")
        if upload_seg is not None:
            seg_path = save_uploaded_file(upload_seg, run_dir, f"{case_id}_seg")
        st.success("Uploaded files are ready for inference.")


run_clicked = st.button("Run Inference", type="primary")
if run_clicked:
    missing_modalities = [m for m in MODALITIES if m not in modality_paths]
    if missing_modalities:
        st.error(f"Missing required modalities: {missing_modalities}")
        st.stop()

    with st.spinner("Preprocessing and running ensemble inference..."):
        input_np, used_slice = build_input_tensor(modality_paths=modality_paths, seg_path=seg_path)
        input_tensor = torch.from_numpy(input_np).unsqueeze(0).to(device)

        result = ensemble_predict(models_bundle["ensemble"], input_tensor)
        idh_target = result.idh_pred
        cam = compute_ensemble_gradcam(
            ensemble_models=models_bundle["ensemble"],
            input_tensor=input_tensor,
            target_head="idh",
            target_class=idh_target,
        )
        base_rgb = make_multimodal_rgb(input_tensor)
        overlay = overlay_heatmap(base_rgb, cam, alpha=0.42)

    st.session_state["last_case_id"] = case_id
    st.session_state["last_slice"] = used_slice
    st.session_state["last_result"] = result
    st.session_state["last_overlay"] = overlay


if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    overlay = st.session_state["last_overlay"]
    case_id = st.session_state["last_case_id"]
    used_slice = st.session_state["last_slice"]

    st.subheader("Predictions")
    c1, c2, c3 = st.columns(3)
    c1.metric("IDH", result.idh_label, f"p={result.idh_prob:.3f}")
    c2.metric("MGMT", result.mgmt_label, f"p={result.mgmt_prob:.3f}")
    c3.metric("Grade", result.grade_label, f"p={result.grade_prob:.3f}")

    st.write(
        f"**Thresholds used:** IDH=`{IDH_THRESHOLD}` (paper final ensemble), "
        f"MGMT=`{MGMT_THRESHOLD}`.  |  **Slice index:** `{used_slice}`"
    )

    st.subheader("GradCAM")
    st.image(overlay, caption="IDH-target GradCAM overlay (T1/T1ce/T2 composite)", use_container_width=True)

    report_name = f"radiogenomics_report_{case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    report_path = REPORTS_DIR / report_name

    if st.button("Generate PDF Report"):
        saved_path = export_pdf_report(
            output_path=report_path,
            patient_id=case_id,
            predictions={
                "idh": result.idh_label,
                "mgmt": result.mgmt_label,
                "grade": result.grade_label,
            },
            probabilities={
                "idh": result.idh_prob,
                "mgmt": result.mgmt_prob,
                "grade": result.grade_prob,
            },
            thresholds={"idh": IDH_THRESHOLD, "mgmt": MGMT_THRESHOLD},
            overlay_rgb=overlay,
            metadata={
                "slice_index": used_slice,
                "seed42_path": models_bundle["paths"]["seed42"],
                "seed2024_path": models_bundle["paths"]["seed2024"],
            },
        )
        st.success(f"Report saved to: `{saved_path}`")
        with open(saved_path, "rb") as f:
            st.download_button(
                label="Download PDF",
                data=f.read(),
                file_name=saved_path.name,
                mime="application/pdf",
            )
