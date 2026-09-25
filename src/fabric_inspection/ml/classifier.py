"""Supervised baseline: ResNet-18 fine-tuned on the six carpet classes, with a CAM output.

The network ends in global average pooling + one linear layer, so Grad-CAM equals CAM up to a
constant (Selvaraju et al., sec. 3.1; docs/investigacion.md #16). The map is therefore exported
as a second ONNX output and serving needs no gradients.
"""

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - conventional alias
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader

from fabric_inspection.domain.catalog import CLASS_CODES
from fabric_inspection.ml.backbones import create_backbone
from fabric_inspection.ml.datasets import FabricDataset, Sample, stack

logger = logging.getLogger(__name__)


class CamClassifier(nn.Module):
    """Returns class probabilities and one class activation map per class."""

    backbone: Any  # timm model: forward_features / forward_head / get_classifier

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def logits_and_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone.forward_features(images)
        logits = self.backbone.forward_head(features)
        return logits, features

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, features = self.logits_and_features(images)
        weights = self.backbone.get_classifier().weight  # (classes, channels)
        cam = torch.einsum("bkhw,ck->bchw", features, weights)
        return torch.softmax(logits, dim=1), cam


def build_classifier(backbone: str, weights_dir: Path | None) -> CamClassifier:
    model = create_backbone(backbone, weights_dir=weights_dir, num_classes=len(CLASS_CODES))
    return CamClassifier(model)


@dataclass
class ClassifierConfig:
    backbone: str = "resnet18"
    epochs: int = 12
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    history: list[dict[str, float]] = field(default_factory=list)


def class_weights(samples: list[Sample]) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1 (few defects, many good images)."""
    counts = np.bincount([sample.label for sample in samples], minlength=len(CLASS_CODES))
    weights = counts.sum() / np.maximum(counts, 1) / len(CLASS_CODES)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


@torch.inference_mode()
def predict_probabilities(
    model: CamClassifier, samples: list[Sample], batch: int = 32
) -> np.ndarray:
    model.eval()
    outputs = [
        model(stack(samples[start : start + batch]))[0].numpy()
        for start in range(0, len(samples), batch)
    ]
    return np.concatenate(outputs) if outputs else np.empty((0, len(CLASS_CODES)))


def defect_scores(probabilities: np.ndarray) -> np.ndarray:
    """Binary defect score of the classifier: 1 - P(good)."""
    return 1.0 - probabilities[:, CLASS_CODES.index("good")]


def train_classifier(
    train: list[Sample],
    val: list[Sample],
    config: ClassifierConfig,
    *,
    weights_dir: Path | None,
    generator: torch.Generator,
) -> CamClassifier:
    """Fine-tune and keep the epoch with the best validation AUPRC (test data is never used)."""
    model = build_classifier(config.backbone, weights_dir)
    loader = DataLoader(
        FabricDataset(train, augment=True, seed=config.seed),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train))
    val_targets = np.array([sample.is_defect for sample in val])
    best_state, best_key = copy.deepcopy(model.state_dict()), (-1.0, -float("inf"))
    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        for images, labels in loader:
            optimizer.zero_grad()
            logits, _ = model.logits_and_features(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        scheduler.step()
        probabilities = predict_probabilities(model, val)
        val_loss = float(
            F.nll_loss(
                torch.log(torch.from_numpy(probabilities).clamp_min(1e-8)),
                torch.tensor([sample.label for sample in val]),
            )
        )
        val_auprc = float(average_precision_score(val_targets, defect_scores(probabilities)))
        config.history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(train),
                "val_loss": val_loss,
                "val_auprc": val_auprc,
            }
        )
        logger.info(
            "epoch %d train_loss=%.4f val_loss=%.4f val_auprc=%.4f",
            epoch,
            total / len(train),
            val_loss,
            val_auprc,
        )
        key = (val_auprc, -val_loss)
        if key > best_key:
            best_key, best_state = key, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model
