"""End-to-end pipeline on a synthetic dataset with random weights (no downloads)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fabric_inspection.data.manifest import read_manifest
from fabric_inspection.ml.classifier import ClassifierConfig, build_classifier
from fabric_inspection.ml.pipeline import PipelineConfig, run_pipeline
from fabric_inspection.registry import ModelRegistry, sha256_file
from tests.synthetic import make_dataset

pytestmark = pytest.mark.slow


def _config(tmp_path: Path, dataset: Path, seed: int = 7) -> PipelineConfig:
    return PipelineConfig(
        processed_dir=tmp_path / "processed",
        manifest_path=tmp_path / "manifest.csv",
        weights_dir=None,
        model_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        registry_path=tmp_path / "models" / "registry.json",
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        image_size=64,
        seed=seed,
        patchcore_backbones=("resnet18",),
        coreset_ratio=0.05,
        classifier=ClassifierConfig(epochs=2, batch_size=8, seed=seed),
        dataset_dir=dataset,
    )


def test_pipeline_produces_registry_metrics_and_verified_models(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "raw")
    config = _config(tmp_path, dataset)
    metrics = run_pipeline(config)

    registry = ModelRegistry.load(config.registry_path)
    for entry in (registry.detector, registry.classifier):
        model_file = config.model_dir / entry.file
        assert sha256_file(model_file) == entry.sha256
    assert registry.detector.threshold is not None
    assert registry.detector.reference_scores
    assert registry.classifier.class_codes is not None

    saved = json.loads((config.reports_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["dataset"]["leakage_pairs"] == 0
    assert set(saved["models"]) == {"classifier_resnet18", "patchcore_resnet18"}
    assert saved["serving"]["onnx_parity_max_abs_diff"]["patchcore"] < 1e-3
    assert 0 <= metrics["serving"]["served_detector_test"]["auprc"] <= 1
    for figure in ("pr_curves", "confusion_classifier", "examples", "score_distribution"):
        assert (config.reports_dir / "figures" / f"{figure}.png").is_file()

    records = read_manifest(config.manifest_path)
    splits = {record.split for record in records}
    assert splits == {"train", "val", "test"}


def test_same_seed_gives_the_same_split_and_scores(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "raw")
    first = run_pipeline(_config(tmp_path / "a", dataset))
    second = run_pipeline(_config(tmp_path / "b", dataset))
    assert first["dataset"]["manifest_sha256"] == second["dataset"]["manifest_sha256"]
    assert (
        first["models"]["patchcore_resnet18"]["test"]["auprc"]
        == second["models"]["patchcore_resnet18"]["test"]["auprc"]
    )
    assert (
        first["models"]["classifier_resnet18"]["test"]["auprc"]
        == second["models"]["classifier_resnet18"]["test"]["auprc"]
    )


def test_exported_cam_equals_grad_cam_up_to_a_constant() -> None:
    """Grad-CAM (autograd) vs the CAM output exported to ONNX (Selvaraju et al., sec. 3.1)."""
    torch.manual_seed(0)
    model = build_classifier("resnet18", None).eval()
    images = torch.randn(1, 3, 64, 64)
    logits, features = model.logits_and_features(images)
    features.retain_grad()
    target = int(logits.argmax())
    logits[0, target].backward()
    assert features.grad is not None
    alpha = features.grad.mean(dim=(2, 3), keepdim=True)
    grad_cam = torch.relu((alpha * features).sum(dim=1))[0].detach().numpy()
    _, cam = model(images)
    height, width = features.shape[-2:]
    expected = torch.relu(cam[0, target]).detach().numpy() / (height * width)
    np.testing.assert_allclose(grad_cam, expected, rtol=1e-4, atol=1e-6)
