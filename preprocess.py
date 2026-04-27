from __future__ import annotations

from pathlib import Path
from typing import Mapping

import cv2
import nibabel as nib
import numpy as np

MODALITIES = ("t1", "t1ce", "t2", "flair")
DEFAULT_SLICE_INDEX = 80


def find_patient_dirs(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _find_modality_file(patient_dir: Path, modality: str) -> Path | None:
    candidates = sorted(patient_dir.glob(f"*_{modality}.nii*"))
    return candidates[0] if candidates else None


def discover_case_files(patient_dir: str | Path) -> tuple[dict[str, Path], Path | None]:
    patient_path = Path(patient_dir)
    modality_paths: dict[str, Path] = {}
    for mod in MODALITIES:
        mod_path = _find_modality_file(patient_path, mod)
        if mod_path is not None:
            modality_paths[mod] = mod_path
    seg_candidates = sorted(patient_path.glob("*_seg.nii*"))
    seg_path = seg_candidates[0] if seg_candidates else None
    return modality_paths, seg_path


def _safe_load_nii(path: str | Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata()


def _zscore_nonzero(slice_2d: np.ndarray) -> np.ndarray:
    out = slice_2d.astype(np.float32, copy=True)
    mask = out > 0
    if np.any(mask):
        values = out[mask]
        out[mask] = (values - values.mean()) / (values.std() + 1e-8)
    return out


def _clip_slice_index(slice_idx: int, depth: int) -> int:
    return max(0, min(int(slice_idx), max(0, depth - 1)))


def _select_slice_from_seg(seg_volume: np.ndarray | None, fallback_slice_idx: int, depth: int) -> int:
    if seg_volume is not None and seg_volume.ndim == 3:
        tumor_slices = np.where(seg_volume.sum(axis=(0, 1)) > 0)[0]
        if len(tumor_slices) > 0:
            # Paper/notebook pipeline: middle slice across tumor extent.
            chosen = int(tumor_slices[len(tumor_slices) // 2])
            return _clip_slice_index(chosen, depth)
    return _clip_slice_index(fallback_slice_idx, depth)


def build_input_tensor(
    modality_paths: Mapping[str, str | Path],
    seg_path: str | Path | None = None,
    slice_idx: int | None = None,
    output_size: int = 224,
    fallback_slice_idx: int = DEFAULT_SLICE_INDEX,
) -> tuple[np.ndarray, int]:
    """
    Returns:
        channels: np.ndarray of shape (4, 224, 224), dtype=float32
        used_slice_idx: int
    """
    missing = [m for m in MODALITIES if m not in modality_paths]
    if missing:
        raise ValueError(f"Missing modalities: {missing}. Expected {MODALITIES}.")

    vols: dict[str, np.ndarray] = {m: _safe_load_nii(modality_paths[m]) for m in MODALITIES}
    first_depth = vols["t1"].shape[2]
    seg_volume = _safe_load_nii(seg_path) if seg_path else None

    if slice_idx is None:
        used_slice_idx = _select_slice_from_seg(seg_volume, fallback_slice_idx, first_depth)
    else:
        used_slice_idx = _clip_slice_index(slice_idx, first_depth)

    channels: list[np.ndarray] = []
    for mod in MODALITIES:
        vol = vols[mod]
        if vol.ndim != 3:
            raise ValueError(f"Modality {mod} has invalid shape {vol.shape}; expected 3D volume.")
        idx = _clip_slice_index(used_slice_idx, vol.shape[2])
        slc = vol[:, :, idx].astype(np.float32)
        slc = _zscore_nonzero(slc)
        slc = cv2.resize(slc, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
        channels.append(slc)

    return np.stack(channels, axis=0).astype(np.float32), used_slice_idx
