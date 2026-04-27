from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _detect_suffix(filename: str) -> str:
    low = filename.lower()
    if low.endswith(".nii.gz"):
        return ".nii.gz"
    if low.endswith(".nii"):
        return ".nii"
    return ".nii"


def save_uploaded_file(uploaded_file: Any, out_dir: str | Path, stem: str) -> Path:
    out_root = ensure_dir(out_dir)
    suffix = _detect_suffix(getattr(uploaded_file, "name", "upload.nii"))
    out_path = out_root / f"{stem}{suffix}"
    out_path.write_bytes(uploaded_file.getvalue())
    return out_path


def export_pdf_report(
    output_path: str | Path,
    patient_id: str,
    predictions: dict[str, str],
    probabilities: dict[str, float],
    thresholds: dict[str, float],
    overlay_rgb: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    out = Path(output_path)
    ensure_dir(out.parent)

    c = canvas.Canvas(str(out), pagesize=A4)
    page_w, page_h = A4

    y = page_h - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Brain Tumor Radiogenomics Report")

    y -= 24
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 14
    c.drawString(40, y, f"Patient/Case: {patient_id}")

    y -= 24
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Predictions")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(48, y, f"IDH: {predictions.get('idh', 'N/A')}  (p={probabilities.get('idh', 0.0):.4f})")
    y -= 14
    c.drawString(48, y, f"MGMT: {predictions.get('mgmt', 'N/A')}  (p={probabilities.get('mgmt', 0.0):.4f})")
    y -= 14
    c.drawString(48, y, f"Grade: {predictions.get('grade', 'N/A')}  (p={probabilities.get('grade', 0.0):.4f})")

    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Thresholds")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(48, y, f"IDH threshold: {thresholds.get('idh', 0.49):.2f} (paper final ensemble)")
    y -= 14
    c.drawString(48, y, f"MGMT threshold: {thresholds.get('mgmt', 0.50):.2f}")

    if metadata:
        y -= 22
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Run Metadata")
        y -= 16
        c.setFont("Helvetica", 10)
        for k, v in metadata.items():
            c.drawString(48, y, f"{k}: {v}")
            y -= 14
            if y < 120:
                break

    if overlay_rgb is not None:
        y_img_top = min(y - 20, page_h - 340)
        if y_img_top < 180:
            c.showPage()
            y_img_top = page_h - 120
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, y_img_top + 20, "GradCAM Overlay")
        else:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, y_img_top + 10, "GradCAM Overlay")

        img = Image.fromarray((np.clip(overlay_rgb, 0, 1) * 255).astype(np.uint8))
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        c.drawImage(ImageReader(buf), 40, y_img_top - 250, width=300, height=250, preserveAspectRatio=True, mask="auto")

    c.showPage()
    c.save()
    return out
