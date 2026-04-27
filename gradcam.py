from __future__ import annotations

import cv2
import numpy as np
import torch

_SEG_MASK_CONTEXT: np.ndarray | None = None


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def set_segmentation_mask_context(seg_mask: np.ndarray | None) -> None:
    """
    Keep API unchanged.
    Expected seg mask shape: (H, W), any non-zero means tumor.
    """
    global _SEG_MASK_CONTEXT
    if seg_mask is None:
        _SEG_MASK_CONTEXT = None
        return
    _SEG_MASK_CONTEXT = (seg_mask > 0).astype(np.float32)


def compute_tumor_attention(seg_mask: np.ndarray) -> np.ndarray:
    """
    Segmentation-guided attention map:
    binary mask -> distance transform -> gaussian smoothing -> normalize
    + slight rim emphasis.
    """
    mask = (seg_mask > 0).astype(np.float32)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.float32)

    dist = cv2.distanceTransform((mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
    dist = dist / (dist.max() + 1e-8)

    attention = cv2.GaussianBlur(dist, (31, 31), 0)

    # Optional mild rim emphasis for enhancing boundary visibility.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rim = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    rim = cv2.GaussianBlur(rim, (15, 15), 0)
    attention = attention + 0.15 * rim

    attention = np.power(attention, 0.8)
    attention = attention / (attention.max() + 1e-8)
    return attention.astype(np.float32)


def make_multimodal_rgb(input_tensor: torch.Tensor) -> np.ndarray:
    """
    Keep API unchanged.
    Uses T1ce as anatomical background (grayscale RGB).
    """
    x = input_tensor.detach().cpu().numpy()[0]
    t1ce = _minmax_norm(x[1])
    return np.stack([t1ce, t1ce, t1ce], axis=-1)


def compute_gradcam(
    model: torch.nn.Module,  # kept for API compatibility
    input_tensor: torch.Tensor,  # kept for API compatibility
    target_head: str = "idh",  # kept for API compatibility
    target_class: int | None = None,  # kept for API compatibility
) -> np.ndarray | None:
    """
    API-compatible single-model explainability.
    Returns segmentation-guided attention if segmentation exists, else None.
    """
    del model, input_tensor, target_head, target_class
    if _SEG_MASK_CONTEXT is None:
        return None
    return compute_tumor_attention(_SEG_MASK_CONTEXT)


def compute_ensemble_gradcam(
    ensemble_models: list[torch.nn.Module],  # kept for API compatibility
    input_tensor: torch.Tensor,  # kept for API compatibility
    target_head: str = "idh",  # kept for API compatibility
    target_class: int | None = None,  # kept for API compatibility
) -> np.ndarray | None:
    """
    API-compatible ensemble explainability.
    Per requirements: return segmentation-guided attention if segmentation exists.
    If segmentation missing, return None.
    """
    del ensemble_models, input_tensor, target_head, target_class
    if _SEG_MASK_CONTEXT is None:
        return None
    return compute_tumor_attention(_SEG_MASK_CONTEXT)


def overlay_heatmap(rgb_image: np.ndarray, heatmap: np.ndarray | None, alpha: float = 0.42) -> np.ndarray:
    """
    Keep API unchanged.
    If heatmap is None, returns anatomical image unchanged.
    """
    base = np.clip(rgb_image, 0.0, 1.0).astype(np.float32)
    if heatmap is None:
        return base

    cam = _minmax_norm(heatmap)
    cam_u8 = np.uint8(cam * 255)
    color = cv2.applyColorMap(cam_u8, cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Keep contiguous hotspot only inside active mask.
    mask = cam > 0.05
    out = base.copy()
    out[mask] = (1.0 - alpha) * base[mask] + alpha * color[mask]
    return np.clip(out, 0.0, 1.0)
