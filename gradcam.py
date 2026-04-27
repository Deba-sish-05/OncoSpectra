from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_SEG_MASK_CONTEXT: np.ndarray | None = None


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def set_segmentation_mask_context(seg_mask: np.ndarray | None) -> None:
    """
    Optional context setter for GradCAM cleanup.
    Expected mask shape: (H, W), values in {0,1} or [0,1].
    """
    global _SEG_MASK_CONTEXT
    if seg_mask is None:
        _SEG_MASK_CONTEXT = None
        return
    _SEG_MASK_CONTEXT = (seg_mask > 0).astype(np.float32)


def _crop_brain_multi(
    x: torch.Tensor,
    threshold: float = 0.05,
    pad: int = 5,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """
    Notebook-aligned crop for multi-channel input.
    Expects x shape (4, H, W).
    Returns cropped tensor and bounding box (y0, y1, x0, x1).
    """
    _, h, w = x.shape
    img = x[0].detach().cpu().numpy()
    img = _minmax_norm(img)
    mask = img > threshold

    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return x, (0, h, 0, w)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    y0 = max(0, int(y0) - pad)
    x0 = max(0, int(x0) - pad)
    y1 = min(h, int(y1) + pad)
    x1 = min(w, int(x1) + pad)

    return x[:, y0:y1, x0:x1], (y0, y1, x0, x1)


def _select_score(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    target_head: str,
    target_class: int | None,
) -> torch.Tensor:
    idh_logit, mgmt_logit, grade_logits = outputs

    if target_head == "idh":
        prob = torch.sigmoid(idh_logit).view(-1)[0]
        cls = int(prob.item() >= 0.5) if target_class is None else int(target_class)
        return prob if cls == 1 else (1.0 - prob)
    if target_head == "mgmt":
        prob = torch.sigmoid(mgmt_logit).view(-1)[0]
        cls = int(prob.item() >= 0.5) if target_class is None else int(target_class)
        return prob if cls == 1 else (1.0 - prob)
    if target_head == "grade":
        cls = int(torch.argmax(grade_logits, dim=1).item()) if target_class is None else int(target_class)
        return grade_logits[0, cls]
    raise ValueError("target_head must be one of: idh, mgmt, grade")


def _gradcam_pp_raw(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    target_head: str,
    target_class: int | None,
) -> np.ndarray:
    """
    Raw GradCAM++ map from last conv layer.
    Target hook layer from notebook: model.backbone.layer4[-1]
    """
    model.eval()

    features: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    target_layer = model.backbone.layer4[-1]

    def fwd_hook(_module, _inp, out):
        features.clear()
        features.append(out)

    def bwd_hook(_module, _grad_in, grad_out):
        gradients.clear()
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        outputs = model(x_batch)
        score = _select_score(outputs, target_head, target_class)

        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        grad = gradients[0]
        fmap = features[0]

        # GradCAM++ weighting (notebook final style).
        g2 = grad ** 2
        g3 = grad ** 3
        eps = 1e-8
        alpha = g2 / (2.0 * g2 + torch.sum(fmap * g3, dim=(2, 3), keepdim=True) + eps)
        weights = torch.sum(alpha * F.relu(grad), dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * fmap, dim=1).squeeze(0)
        cam = F.relu(cam)
        cam_np = cam.detach().cpu().numpy()
        return _minmax_norm(cam_np)
    finally:
        h1.remove()
        h2.remove()


def _keep_largest_region(cam: np.ndarray) -> np.ndarray:
    binary = (cam > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return cam

    candidates: list[tuple[float, int]] = []
    for lbl in range(1, num_labels):
        area = float(stats[lbl, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        bbox_area = float(max(1, w * h))
        compactness = area / bbox_area

        region_vals = cam[labels == lbl]
        mean_act = float(region_vals.mean()) if region_vals.size > 0 else 0.0

        # Strength favors compact high-activation regions.
        score = mean_act * area * compactness
        candidates.append((score, lbl))

    if not candidates:
        return cam

    candidates.sort(reverse=True, key=lambda t: t[0])
    keep_labels = [candidates[0][1]]

    # Keep top-2 only when second region is competitively strong.
    if len(candidates) > 1 and candidates[1][0] >= 0.55 * candidates[0][0]:
        keep_labels.append(candidates[1][1])

    keep_mask = np.isin(labels, keep_labels).astype(np.float32)
    return cam * keep_mask


def _clean_cam(
    cam: np.ndarray,
    anatomical_img: np.ndarray,
    seg_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Notebook-aligned post-processing:
    - resize
    - Gaussian smoothing
    - 93rd percentile hotspot threshold
    - morphology opening/closing to remove speckles
    - top compact connected region filtering
    - optional segmentation constraint
    - brain masking
    """
    h, w = anatomical_img.shape
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    cam = cv2.GaussianBlur(cam, (15, 15), 0)

    th = np.percentile(cam, 93)
    cam = np.where(cam >= th, cam, 0).astype(np.float32)

    # Morphological cleanup (denoise and fill tiny holes).
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cam_bin = (cam > 0).astype(np.uint8)
    cam_bin = cv2.morphologyEx(cam_bin, cv2.MORPH_OPEN, k, iterations=1)
    cam_bin = cv2.morphologyEx(cam_bin, cv2.MORPH_CLOSE, k, iterations=1)
    cam = cam * cam_bin.astype(np.float32)

    cam = _keep_largest_region(cam)

    if seg_mask is not None:
        seg_r = cv2.resize(seg_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        cam = cam * (seg_r > 0).astype(np.float32)

    # Brain mask from anatomical slice (T1ce-like channel in notebook).
    mask = (anatomical_img > np.percentile(anatomical_img, 20)).astype(np.float32)
    cam = cam * mask

    # Slight edge smoothing after masking.
    cam = cv2.GaussianBlur(cam, (5, 5), 0)
    return _minmax_norm(cam)


def make_multimodal_rgb(input_tensor: torch.Tensor) -> np.ndarray:
    """
    API preserved.
    Returns an anatomical RGB image (T1ce repeated into 3 channels)
    for clean medical overlay visualization.
    Expects input tensor shape (1, 4, H, W).
    """
    x = input_tensor.detach().cpu().numpy()[0]
    t1ce = _minmax_norm(x[1])
    return np.stack([t1ce, t1ce, t1ce], axis=-1)


def compute_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_head: str = "idh",
    target_class: int | None = None,
) -> np.ndarray:
    """
    API preserved.
    Returns cleaned full-size CAM (H, W), localized to tumor hotspot.
    """
    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError("compute_gradcam expects input_tensor shape (1, 4, H, W)")

    _, _, h, w = input_tensor.shape
    x = input_tensor[0]

    x_crop, (y0, y1, x0, x1) = _crop_brain_multi(x)
    x_crop_batch = x_crop.unsqueeze(0)

    raw_cam = _gradcam_pp_raw(
        model=model,
        x_batch=x_crop_batch,
        target_head=target_head,
        target_class=target_class,
    )

    # Notebook uses channel 1 (T1ce) as anatomical guide for CAM cleanup.
    img_ref = x_crop[1].detach().cpu().numpy()
    img_ref = _minmax_norm(img_ref)
    seg_crop = None
    if _SEG_MASK_CONTEXT is not None:
        seg_crop = _SEG_MASK_CONTEXT[y0:y1, x0:x1]
    cam_crop = _clean_cam(raw_cam, img_ref, seg_mask=seg_crop)

    full_cam = np.zeros((h, w), dtype=np.float32)
    region_w = max(1, x1 - x0)
    region_h = max(1, y1 - y0)
    cam_region = cv2.resize(cam_crop, (region_w, region_h), interpolation=cv2.INTER_LINEAR)
    full_cam[y0:y1, x0:x1] = cam_region
    return _minmax_norm(full_cam)


def compute_ensemble_gradcam(
    ensemble_models: list[torch.nn.Module],
    input_tensor: torch.Tensor,
    target_head: str = "idh",
    target_class: int | None = None,
) -> np.ndarray:
    """
    API preserved.
    Ensemble CAM = average of cleaned per-model CAMs.
    """
    cams = [
        compute_gradcam(
            model=model,
            input_tensor=input_tensor,
            target_head=target_head,
            target_class=target_class,
        )
        for model in ensemble_models
    ]
    return _minmax_norm(np.mean(np.stack(cams, axis=0), axis=0))


def overlay_heatmap(rgb_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    """
    API preserved.
    Overlay hotspot heatmap only where activation exists to avoid diffuse full-image artifact.
    """
    base = np.clip(rgb_image, 0.0, 1.0).astype(np.float32)
    cam = _minmax_norm(heatmap)

    cam_u8 = np.uint8(cam * 255)
    color = cv2.applyColorMap(cam_u8, cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    out = base.copy()
    mask = cam > 0
    out[mask] = (1.0 - alpha) * base[mask] + alpha * color[mask]
    return np.clip(out, 0.0, 1.0)
