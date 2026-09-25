"""PatchCore (Roth et al., CVPR 2022; docs/investigacion.md #15), written to export as one ONNX.

Training uses only defect-free images: mid-level features of an ImageNet backbone are pooled
over a 3x3 neighbourhood, a greedy coreset keeps a small representative memory bank, and the
anomaly score of a patch is the distance to its nearest neighbour in that bank.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - conventional alias
from torch import nn

from fabric_inspection.ml.backbones import create_backbone
from fabric_inspection.ml.datasets import Sample, stack

logger = logging.getLogger(__name__)


def gaussian_kernel(sigma: float) -> torch.Tensor:
    radius = math.ceil(4 * sigma)
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    line = torch.exp(-(coords**2) / (2 * sigma**2))
    line = line / line.sum()
    return (line[:, None] * line[None, :])[None, None]


class PatchEmbedder(nn.Module):
    """Layer-2 and layer-3 features, 3x3 local average, layer 3 upsampled to layer-2 size."""

    def __init__(self, extractor: nn.Module) -> None:
        super().__init__()
        self.extractor = extractor

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        low, high = self.extractor(images)
        low = F.avg_pool2d(low, kernel_size=3, stride=1, padding=1)
        high = F.avg_pool2d(high, kernel_size=3, stride=1, padding=1)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([low, high], dim=1)


class PatchCore(nn.Module):
    """images (B,3,S,S) -> (image score (B,), anomaly map (B,S,S))."""

    memory_bank: torch.Tensor
    memory_sq: torch.Tensor
    kernel: torch.Tensor

    def __init__(
        self,
        embedder: PatchEmbedder,
        memory_bank: torch.Tensor,
        image_size: int,
        sigma: float = 4.0,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.image_size = image_size
        self.register_buffer("memory_bank", memory_bank.contiguous())
        self.register_buffer("memory_sq", (memory_bank**2).sum(dim=1))
        kernel = gaussian_kernel(sigma)
        self.register_buffer("kernel", kernel)
        self.pad = int(kernel.shape[-1]) // 2

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.embedder(images)
        batch, dim, height, width = embedding.shape
        patches = embedding.permute(0, 2, 3, 1).reshape(batch, height * width, dim)
        squared = (
            (patches**2).sum(dim=-1, keepdim=True)
            - 2 * patches @ self.memory_bank.T
            + self.memory_sq
        )
        nearest = squared.min(dim=-1).values.clamp_min(0).sqrt()
        patch_map = nearest.reshape(batch, 1, height, width)
        size = (self.image_size, self.image_size)
        anomaly = F.interpolate(patch_map, size=size, mode="bilinear", align_corners=False)
        anomaly = F.conv2d(F.pad(anomaly, [self.pad] * 4, mode="reflect"), self.kernel)
        return anomaly.amax(dim=(1, 2, 3)), anomaly[:, 0]


def greedy_coreset(
    features: torch.Tensor, ratio: float, seed: int, projection_dim: int = 128
) -> torch.Tensor:
    """Greedy k-center selection on a seeded random projection (Johnson-Lindenstrauss)."""
    total = features.shape[0]
    count = max(1, int(total * ratio))
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(features.shape[1], projection_dim, generator=generator)
    reduced = features @ projection
    selected = [int(torch.randint(total, (1,), generator=generator))]
    min_distance = ((reduced - reduced[selected[0]]) ** 2).sum(dim=1)
    for _ in range(count - 1):
        index = int(torch.argmax(min_distance))
        selected.append(index)
        min_distance = torch.minimum(min_distance, ((reduced - reduced[index]) ** 2).sum(dim=1))
    return features[torch.tensor(selected)]


@dataclass
class PatchCoreConfig:
    backbone: str = "wide_resnet50_2"
    coreset_ratio: float = 0.01
    sigma: float = 4.0
    seed: int = 42
    batch_size: int = 8


def build_embedder(backbone: str, weights_dir: Path | None) -> PatchEmbedder:
    extractor = create_backbone(
        backbone, weights_dir=weights_dir, features_only=True, out_indices=(2, 3)
    )
    return PatchEmbedder(extractor).eval()


@torch.inference_mode()
def fit_patchcore(
    normal: list[Sample], image_size: int, config: PatchCoreConfig, *, weights_dir: Path | None
) -> PatchCore:
    embedder = build_embedder(config.backbone, weights_dir)
    chunks = []
    for start in range(0, len(normal), config.batch_size):
        embedding = embedder(stack(normal[start : start + config.batch_size]))
        chunks.append(embedding.permute(0, 2, 3, 1).reshape(-1, embedding.shape[1]))
    features = torch.cat(chunks)
    logger.info("PatchCore: %d patch features of dim %d", *features.shape)
    memory = greedy_coreset(features, config.coreset_ratio, config.seed)
    logger.info("PatchCore: memory bank of %d patches", memory.shape[0])
    return PatchCore(embedder, memory.clone(), image_size, config.sigma).eval()


@torch.inference_mode()
def score_samples(
    model: nn.Module, samples: list[Sample], batch: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Image scores (N,) and anomaly maps (N,S,S) of a PatchCore module."""
    model.eval()
    scores, maps = [], []
    for start in range(0, len(samples), batch):
        score, anomaly = model(stack(samples[start : start + batch]))
        scores.append(score.numpy())
        maps.append(anomaly.numpy())
    return np.concatenate(scores), np.concatenate(maps)
