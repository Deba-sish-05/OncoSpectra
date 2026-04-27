from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models


class RadiogenomicsModel(nn.Module):
    """
    Final ResNet50-v2 multi-task architecture (paper-aligned).

    Input:
        (B, 4, 224, 224) ordered as [T1, T1ce, T2, FLAIR]
    Outputs:
        idh_logits:   (B, 1)
        mgmt_logits:  (B, 1)
        grade_logits: (B, 2)
    """

    def __init__(self) -> None:
        super().__init__()

        self.backbone = models.resnet50(weights=None)

        # Replace first conv to accept 4 MRI channels.
        orig_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            4,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        with torch.no_grad():
            self.backbone.conv1.weight[:, :3] = orig_conv.weight
            self.backbone.conv1.weight[:, 3] = orig_conv.weight.mean(dim=1)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # ResNet-50 (v2) fusion + heads (dropout=0.45 from paper/table).
        self.fusion = nn.Sequential(
            nn.Dropout(0.45),
            nn.Linear(in_features, 512),
            nn.ReLU(),
        )

        self.idh_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(256, 1),
        )
        self.mgmt_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(256, 1),
        )
        self.grade_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        fused = self.fusion(features)
        return self.idh_head(fused), self.mgmt_head(fused), self.grade_head(fused)


def build_resnet50_v2_model(device: torch.device | str) -> RadiogenomicsModel:
    model = RadiogenomicsModel()
    model.to(device)
    return model


def load_checkpoint_state(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device | str,
) -> nn.Module:
    """
    Loads checkpoints that are either:
    - {'model': state_dict, ...}
    - raw state_dict
    """
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model
