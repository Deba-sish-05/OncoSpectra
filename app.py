from __future__ import annotations

from datetime import datetime
from pathlib import Path
import uuid

import nibabel as nib
import numpy as np
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


def _prediction_confidence(prob: float, binary: bool = True) -> float:
    if binary:
        return float(max(prob, 1.0 - prob))
    return float(prob)


def _clinical_interpretation(idh: str, mgmt: str, grade: str) -> str:
    if "High Grade" in grade and idh == "Wildtype":
        return "Profile suggests a biologically aggressive glioma pattern; correlate with pathology and treatment planning."
    if "Low Grade" in grade and idh == "Mutant":
        return "Profile suggests a less aggressive glioma phenotype; imaging-genomic pattern is generally favorable."
    if mgmt == "Methylated":
        return "MGMT methylation-positive prediction may indicate better alkylating-agent responsiveness; confirm with molecular testing."
    return "Mixed-risk imaging-genomic profile; integrate with histology, molecular assays, and multidisciplinary review."


def _safe_mid_slice(path: Path, slice_idx: int | None = None) -> np.ndarray:
    vol = nib.load(str(path)).get_fdata()
    if vol.ndim != 3:
        raise ValueError(f"Unexpected volume shape for {path.name}: {vol.shape}")
    z = int(vol.shape[2] // 2) if slice_idx is None else max(0, min(int(slice_idx), vol.shape[2] - 1))
    return vol[:, :, z].astype(np.float32)


def _normalize_for_display(img: np.ndarray) -> np.ndarray:
    mn = float(np.min(img))
    mx = float(np.max(img))
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - mn) / (mx - mn)).astype(np.float32)


def _resolve_preview_slice(modality_paths: dict[str, Path], seg_path: Path | None) -> int:
    if seg_path is not None and seg_path.exists():
        seg = nib.load(str(seg_path)).get_fdata()
        if seg.ndim == 3:
            tumor = np.where(seg.sum(axis=(0, 1)) > 0)[0]
            if len(tumor) > 0:
                return int(tumor[len(tumor) // 2])
    t1 = modality_paths.get("t1")
    if t1 is not None and t1.exists():
        d = nib.load(str(t1)).shape[2]
        return int(d // 2)
    return 0


def _render_modality_previews(modality_paths: dict[str, Path], seg_path: Path | None) -> None:
    if any(m not in modality_paths for m in MODALITIES):
        return

    st.markdown("#### MRI Preview")
    preview_slice = _resolve_preview_slice(modality_paths, seg_path)

    cols = st.columns(5)
    for i, mod in enumerate(MODALITIES):
        sl = _safe_mid_slice(modality_paths[mod], preview_slice)
        cols[i].image(_normalize_for_display(sl), caption=f"{mod.upper()} (slice {preview_slice})", use_container_width=True)

    with cols[4]:
        if seg_path is not None and seg_path.exists():
            seg = _safe_mid_slice(seg_path, preview_slice)
            st.image(_normalize_for_display(seg), caption="SEG Mask", use_container_width=True)
        else:
            st.info("No SEG")


st.set_page_config(page_title="Radiogenomics Dashboard", layout="wide")
st.markdown(
    """
    <style>
      .card {
        border: 1px solid #e6e9ef;
        border-radius: 12px;
        padding: 14px 16px;
        background: #ffffff;
      }
      .card-title {
        font-size: 0.86rem;
        font-weight: 700;
        color: #344054;
        margin-bottom: 8px;
      }
      .card-value {
        font-size: 1.12rem;
        font-weight: 700;
        color: #111827;
      }
      .muted {
        color: #667085;
        font-size: 0.82rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Brain Tumor Radiogenomics Dashboard")
st.caption(
    "Final paper-aligned deployment: ResNet50-v2 multi-task backbone, 2-seed ensemble + horizontal flip TTA, "
    "IDH threshold = 0.49, GradCAM++ on layer4[-1]."
)


@st.cache_resource(show_spinner=False)
def _cached_models(weights_dir: str, device_name: str):
    return load_deployment_models(weights_dir=weights_dir, device=torch.device(device_name))


device = resolve_device()

top_left, top_right = st.columns([1.2, 2.2])
with top_left:
    st.success(f"Runtime Device: {device}")
with top_right:
    ckpt_names = sorted(p.name for p in WEIGHTS_DIR.glob("*.pth"))
    st.info("Checkpoints: " + (", ".join(ckpt_names) if ckpt_names else "None found in weights/"))

try:
    models_bundle = _cached_models(str(WEIGHTS_DIR), str(device))
except Exception as exc:
    st.error(f"Model load failed: {exc}")
    st.stop()

loaded_seed42 = Path(models_bundle["paths"]["seed42"]).name
loaded_seed2024 = Path(models_bundle["paths"]["seed2024"]).name
loaded_single = Path(models_bundle["paths"]["single"]).name
st.caption(f"Loaded: seed42={loaded_seed42}, seed2024={loaded_seed2024}, single={loaded_single}")

mode = st.radio("Input Source", ["Use sample from data/", "Upload NIfTI files"], horizontal=True)

case_id = "unknown_case"
modality_paths: dict[str, Path] = {}
seg_path: Path | None = None

if mode == "Use sample from data/":
    patient_dirs = [p for p in find_patient_dirs(DATA_DIR) if p.name != "_uploads"]
    if not patient_dirs:
        st.warning("No patient folders found in data/.")
    else:
        selected = st.selectbox("Select Patient Folder", options=patient_dirs, format_func=lambda p: p.name)
        case_id = selected.name
        modality_paths, seg_path = discover_case_files(selected)
        missing = [m for m in MODALITIES if m not in modality_paths]
        if missing:
            st.error(f"Missing modalities in selected case: {missing}")
        else:
            st.success(f"Case ready: {case_id}")
else:
    st.markdown("Upload all four modalities (`T1`, `T1ce`, `T2`, `FLAIR`). Segmentation mask is optional.")
    uploads: dict[str, object] = {}
    c1, c2 = st.columns(2)
    with c1:
        uploads["t1"] = st.file_uploader("T1", type=["nii", "gz"], key="up_t1")
        uploads["t2"] = st.file_uploader("T2", type=["nii", "gz"], key="up_t2")
    with c2:
        uploads["t1ce"] = st.file_uploader("T1ce", type=["nii", "gz"], key="up_t1ce")
        uploads["flair"] = st.file_uploader("FLAIR", type=["nii", "gz"], key="up_flair")
    upload_seg = st.file_uploader("Segmentation (optional)", type=["nii", "gz"], key="up_seg")

    if all(uploads.get(m) is not None for m in MODALITIES):
        if "upload_run_id" not in st.session_state:
            st.session_state["upload_run_id"] = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        run_id = st.session_state["upload_run_id"]
        run_dir = ensure_dir(UPLOAD_DIR / run_id)
        case_id = f"upload_{run_id}"
        for mod in MODALITIES:
            modality_paths[mod] = save_uploaded_file(uploads[mod], run_dir, f"{case_id}_{mod}")
        if upload_seg is not None:
            seg_path = save_uploaded_file(upload_seg, run_dir, f"{case_id}_seg")
        st.success("Uploaded files validated and ready.")

if all(m in modality_paths for m in MODALITIES):
    _render_modality_previews(modality_paths, seg_path)

run_clicked = st.button("Run Inference", type="primary")
if run_clicked:
    missing_modalities = [m for m in MODALITIES if m not in modality_paths]
    if missing_modalities:
        st.error(f"Missing required modalities: {missing_modalities}")
        st.stop()

    with st.spinner("Running preprocessing, ensemble inference, and GradCAM++..."):
        input_np, used_slice = build_input_tensor(modality_paths=modality_paths, seg_path=seg_path)
        input_tensor = torch.from_numpy(input_np).unsqueeze(0).to(device)

        # Backend inference logic preserved.
        result = ensemble_predict(models_bundle["ensemble"], input_tensor)
        idh_target = result.idh_pred

        cam = compute_ensemble_gradcam(
            ensemble_models=models_bundle["ensemble"],
            input_tensor=input_tensor,
            target_head="idh",
            target_class=idh_target,
        )
        base_rgb = make_multimodal_rgb(input_tensor)
        overlay = overlay_heatmap(base_rgb, cam, alpha=0.45)

    st.session_state["last_case_id"] = case_id
    st.session_state["last_slice"] = used_slice
    st.session_state["last_result"] = result
    st.session_state["last_overlay"] = overlay
    st.session_state["last_base"] = base_rgb
    st.session_state["last_cam"] = cam
    st.success("Inference complete.")

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    overlay = st.session_state["last_overlay"]
    base_rgb = st.session_state["last_base"]
    case_id = st.session_state["last_case_id"]
    used_slice = st.session_state["last_slice"]

    idh_pct = result.idh_prob * 100.0
    mgmt_pct = result.mgmt_prob * 100.0
    grade_pct = result.grade_prob * 100.0

    idh_conf = _prediction_confidence(result.idh_prob, binary=True) * 100.0
    mgmt_conf = _prediction_confidence(result.mgmt_prob, binary=True) * 100.0
    grade_conf = _prediction_confidence(result.grade_prob, binary=False) * 100.0

    st.markdown("### Prediction Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"<div class='card'><div class='card-title'>IDH</div><div class='card-value'>{result.idh_label}</div>"
            f"<div class='muted'>Probability: {idh_pct:.1f}%</div></div>",
            unsafe_allow_html=True,
        )
        st.progress(int(round(idh_conf)))
        st.caption(f"Confidence: {idh_conf:.1f}%")

    with col2:
        st.markdown(
            f"<div class='card'><div class='card-title'>MGMT</div><div class='card-value'>{result.mgmt_label}</div>"
            f"<div class='muted'>Probability: {mgmt_pct:.1f}%</div></div>",
            unsafe_allow_html=True,
        )
        st.progress(int(round(mgmt_conf)))
        st.caption(f"Confidence: {mgmt_conf:.1f}%")

    with col3:
        st.markdown(
            f"<div class='card'><div class='card-title'>Tumor Grade</div><div class='card-value'>{result.grade_label}</div>"
            f"<div class='muted'>Probability: {grade_pct:.1f}%</div></div>",
            unsafe_allow_html=True,
        )
        st.progress(int(round(grade_conf)))
        st.caption(f"Confidence: {grade_conf:.1f}%")

    st.caption(
        f"Case: {case_id} | Slice Index: {used_slice} | Thresholds: IDH={IDH_THRESHOLD:.2f}, MGMT={MGMT_THRESHOLD:.2f}"
    )

    st.markdown("### GradCAM++")
    g1, g2 = st.columns(2)
    g1.image(base_rgb, caption="Anatomical Slice (T1ce)", use_container_width=True)
    g2.image(overlay, caption="Localized GradCAM++ Hotspot Overlay", use_container_width=True)

    report_name = f"radiogenomics_report_{case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    report_path = REPORTS_DIR / report_name

    if st.button("Generate PDF Report"):
        interpretation = _clinical_interpretation(result.idh_label, result.mgmt_label, result.grade_label)

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
                "seed42_ckpt": loaded_seed42,
                "seed2024_ckpt": loaded_seed2024,
                "single_ckpt": loaded_single,
                "inference_mode": "2-seed ensemble + flip TTA",
                "interpretation": interpretation,
            },
        )
        st.success(f"Report saved: {saved_path.name}")
        with open(saved_path, "rb") as f:
            st.download_button(
                label="Download PDF",
                data=f.read(),
                file_name=saved_path.name,
                mime="application/pdf",
            )
