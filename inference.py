from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from model import build_resnet50_v2_model, load_checkpoint_state

IDH_THRESHOLD = 0.49
MGMT_THRESHOLD = 0.50

IDH_LABELS = {0: "Wildtype", 1: "Mutant"}
MGMT_LABELS = {0: "Unmethylated", 1: "Methylated"}
GRADE_LABELS = {0: "Low Grade (LGG)", 1: "High Grade (GBM)"}


@dataclass
class PredictionResult:
    idh_prob: float
    mgmt_prob: float
    grade_prob: float
    idh_pred: int
    mgmt_pred: int
    grade_pred: int
    idh_label: str
    mgmt_label: str
    grade_label: str


def _first_existing(weights_dir: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        path = weights_dir / name
        if path.exists():
            return path
    return None


def load_deployment_models(weights_dir: str | Path, device: torch.device) -> dict[str, object]:
    """
    Priority from paper deployment:
      1) best_resnet50_seed42.pth + best_resnet50_seed2024.pth (ensemble)
      2) best_resnet50_v2.pth (single/fallback)
    """
    root = Path(weights_dir)
    seed42_path = _first_existing(root, ["best_resnet50_seed42.pth", "ensemble1.pth"])
    seed2024_path = _first_existing(root, ["best_resnet50_seed2024.pth", "ensemble2.pth"])
    v2_path = _first_existing(root, ["best_resnet50_v2.pth", "resnet50_v2.pth"])

    if seed42_path is None or seed2024_path is None:
        if v2_path is None:
            raise FileNotFoundError(
                f"Could not find ensemble checkpoints in {root}. "
                "Expected best_resnet50_seed42.pth and best_resnet50_seed2024.pth."
            )
        # Fallback: reuse v2 model for both slots.
        seed42_path = v2_path
        seed2024_path = v2_path

    model1 = build_resnet50_v2_model(device)
    model2 = build_resnet50_v2_model(device)
    load_checkpoint_state(model1, seed42_path, device)
    load_checkpoint_state(model2, seed2024_path, device)

    single_model = build_resnet50_v2_model(device)
    if v2_path is not None:
        load_checkpoint_state(single_model, v2_path, device)
    else:
        # fallback to first ensemble model checkpoint
        load_checkpoint_state(single_model, seed42_path, device)

    return {
        "ensemble": [model1, model2],
        "single": single_model,
        "paths": {
            "seed42": str(seed42_path),
            "seed2024": str(seed2024_path),
            "single": str(v2_path) if v2_path is not None else str(seed42_path),
        },
    }


def _tta_forward_probs(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Notebook/paper-aligned TTA:
    - forward on original
    - forward on horizontal flip
    - average probabilities
    """
    with torch.no_grad():
        idh_o, mgmt_o, grade_o = model(x)
        x_flip = torch.flip(x, dims=[3])
        idh_f, mgmt_f, grade_f = model(x_flip)

        idh_p = (torch.sigmoid(idh_o) + torch.sigmoid(idh_f)) / 2.0
        mgmt_p = (torch.sigmoid(mgmt_o) + torch.sigmoid(mgmt_f)) / 2.0
        grade_p = (torch.softmax(grade_o, dim=1) + torch.softmax(grade_f, dim=1)) / 2.0
    return idh_p, mgmt_p, grade_p


def ensemble_predict(
    ensemble_models: list[torch.nn.Module],
    x: torch.Tensor,
    idh_threshold: float = IDH_THRESHOLD,
    mgmt_threshold: float = MGMT_THRESHOLD,
) -> PredictionResult:
    if len(ensemble_models) == 0:
        raise ValueError("No models provided for ensemble prediction.")

    idh_probs = []
    mgmt_probs = []
    grade_probs = []

    for model in ensemble_models:
        model.eval()
        p_idh, p_mgmt, p_grade = _tta_forward_probs(model, x)
        idh_probs.append(p_idh)
        mgmt_probs.append(p_mgmt)
        grade_probs.append(p_grade)

    idh_prob_t = torch.mean(torch.stack(idh_probs, dim=0), dim=0).view(-1)[0]
    mgmt_prob_t = torch.mean(torch.stack(mgmt_probs, dim=0), dim=0).view(-1)[0]
    grade_prob_vec = torch.mean(torch.stack(grade_probs, dim=0), dim=0)[0]

    idh_prob = float(idh_prob_t.item())
    mgmt_prob = float(mgmt_prob_t.item())
    grade_pred = int(torch.argmax(grade_prob_vec).item())
    grade_prob = float(grade_prob_vec[grade_pred].item())

    idh_pred = int(idh_prob >= idh_threshold)
    mgmt_pred = int(mgmt_prob >= mgmt_threshold)

    return PredictionResult(
        idh_prob=idh_prob,
        mgmt_prob=mgmt_prob,
        grade_prob=grade_prob,
        idh_pred=idh_pred,
        mgmt_pred=mgmt_pred,
        grade_pred=grade_pred,
        idh_label=IDH_LABELS[idh_pred],
        mgmt_label=MGMT_LABELS[mgmt_pred],
        grade_label=GRADE_LABELS[grade_pred],
    )
