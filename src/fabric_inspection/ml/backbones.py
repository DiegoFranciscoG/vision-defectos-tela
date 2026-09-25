"""ImageNet backbones from timm with weights pinned by revision and SHA-256 (OWASP ML06)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import timm
from torch import nn

from fabric_inspection.data.download import RemoteFile, fetch

HF = "https://huggingface.co/timm"


@dataclass(frozen=True, slots=True)
class Backbone:
    timm_name: str
    weights: RemoteFile
    license: str


# Weights are safetensors (no pickle). Licenses from the Hugging Face model cards (#23).
BACKBONES: dict[str, Backbone] = {
    "resnet18": Backbone(
        timm_name="resnet18",
        weights=RemoteFile(
            url=f"{HF}/resnet18.tv_in1k/resolve/bbd144b3e5565108aad885f145491d11bc6ce807/model.safetensors",
            sha256="694f673df6520a3158624e8a89af086f59923ee4cd7436fe5bc3bc71d295ad81",
            filename="resnet18.tv_in1k.safetensors",
        ),
        license="BSD-3-Clause",
    ),
    "wide_resnet50_2": Backbone(
        timm_name="wide_resnet50_2",
        weights=RemoteFile(
            url=f"{HF}/wide_resnet50_2.tv_in1k/resolve/343c5fa59ca8a131e0cdf42144a641b36a0387c2/model.safetensors",
            sha256="df6fb6c4824769769de18e14088475fd6ee94236849aa4e5d8022ba9d9a9a16c",
            filename="wide_resnet50_2.tv_in1k.safetensors",
        ),
        license="BSD-3-Clause",
    ),
}


def create_backbone(
    name: str,
    *,
    weights_dir: Path | None,
    features_only: bool = False,
    out_indices: tuple[int, ...] = (2, 3),
    num_classes: int = 0,
) -> nn.Module:
    """Build the timm model; `weights_dir=None` gives random weights (used by fast tests)."""
    spec = BACKBONES[name]
    kwargs: dict[str, Any] = {}
    if weights_dir is not None:
        weights = fetch(spec.weights, weights_dir)
        kwargs = {"pretrained": True, "pretrained_cfg_overlay": {"file": str(weights)}}
    if features_only:
        kwargs |= {"features_only": True, "out_indices": out_indices}
    else:
        kwargs["num_classes"] = num_classes
    model: nn.Module = timm.create_model(spec.timm_name, **kwargs)
    return model
