from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def make_multimodal_rgb(input_tensor: torch.Tensor) -> np.ndarray:
    """
    Builds a paper-aligned visualization base using T1, T1ce, T2 channels.
    Expects input tensor shape (1,4,H,W).
    Returns RGB float image in [0,1].
    """
    x = input_tensor.detach().cpu().numpy()[0]
    rgb = np.stack(
        [
            _minmax_norm(x[0]),  # T1
            _minmax_norm(x[1]),  # T1ce
            _minmax_norm(x[2]),  # T2
        ],
        axis=-1,
    )
    return rgb


def compute_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_head: str = "idh",
    target_class: int | None = None,
) -> np.ndarray:
    """
    GradCAM from last convolutional layer of ResNet50:
    model.backbone.layer4[-1]
    """
    model.eval()
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    target_layer = model.backbone.layer4[-1]

    def fwd_hook(_module, _inp, output):
        activations.clear()
        activations.append(output)

    def bwd_hook(_module, _grad_in, grad_out):
        gradients.clear()
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        outputs = model(input_tensor)
        idh_logit, mgmt_logit, grade_logits = outputs

        if target_head == "idh":
            cls = 1 if target_class is None else int(target_class)
            score = idh_logit.view(-1)[0] if cls == 1 else -idh_logit.view(-1)[0]
        elif target_head == "mgmt":
            cls = 1 if target_class is None else int(target_class)
            score = mgmt_logit.view(-1)[0] if cls == 1 else -mgmt_logit.view(-1)[0]
        elif target_head == "grade":
            cls = int(torch.argmax(grade_logits, dim=1).item()) if target_class is None else int(target_class)
            score = grade_logits[0, cls]
        else:
            raise ValueError(f"Unsupported target_head={target_head}.")

        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        grad = gradients[0]
        fmap = activations[0]
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * fmap, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = _minmax_norm(cam)
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]), interpolation=cv2.INTER_LINEAR)
        return _minmax_norm(cam)
    finally:
        h1.remove()
        h2.remove()


def compute_ensemble_gradcam(
    ensemble_models: list[torch.nn.Module],
    input_tensor: torch.Tensor,
    target_head: str = "idh",
    target_class: int | None = None,
) -> np.ndarray:
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
    base = np.clip(rgb_image, 0.0, 1.0)
    cam_u8 = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255)
    color = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out = (1.0 - alpha) * base + alpha * color
    return np.clip(out, 0.0, 1.0)
