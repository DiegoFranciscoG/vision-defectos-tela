"""Tiny, real ONNX models (random weights, 64 px) so API tests exercise the full stack."""

import io
import secrets
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from fabric_inspection.api.app import create_app
from fabric_inspection.config import ApiSettings
from fabric_inspection.db.migrate import upgrade_database
from fabric_inspection.db.session import build_engine, build_session_factory, session_scope
from fabric_inspection.inference.inspector import FabricInspector
from fabric_inspection.ml.classifier import build_classifier
from fabric_inspection.ml.export import export_onnx
from fabric_inspection.ml.patchcore import PatchCore, build_embedder
from fabric_inspection.preprocessing import preprocess
from fabric_inspection.registry import ModelRegistry, sha256_file
from fabric_inspection.service.demo_seed import seed_demo
from tests.conftest import make_registry
from tests.synthetic import make_image

# Generated per test session: no key is ever written in the repository.
API_KEY = secrets.token_urlsafe(32)
SIZE = 64


def png_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModelRegistry, Path]:
    torch.manual_seed(0)
    model_dir = tmp_path_factory.mktemp("models")
    rng = np.random.default_rng(1)
    good = [make_image(rng, SIZE, None)[0] for _ in range(6)]
    embedder = build_embedder("resnet18", None)
    with torch.inference_mode():
        batch = torch.from_numpy(np.concatenate([preprocess(image, SIZE) for image in good]))
        embedding = embedder(batch)
        memory = embedding.permute(0, 2, 3, 1).reshape(-1, embedding.shape[1])
        detector = PatchCore(embedder, memory.clone(), SIZE).eval()
        good_scores = detector(batch)[0].numpy()
    sample = batch[:1]
    export_onnx(detector, sample, model_dir / "detector.onnx", ["score", "anomaly_map"])
    classifier = build_classifier("resnet18", None).eval()
    export_onnx(classifier, sample, model_dir / "classifier.onnx", ["probabilities", "cam"])
    threshold = float(good_scores.max() * 1.05) + 1e-3
    registry = make_registry(
        sha256_file(model_dir / "detector.onnx"),
        sha256_file(model_dir / "classifier.onnx"),
        image_size=SIZE,
    )
    registry.detector.threshold = threshold
    registry.detector.pixel_threshold = threshold * 0.8
    registry.detector.reference_scores = [
        round(threshold * v, 5) for v in np.linspace(0.3, 0.95, 60).tolist()
    ]
    registry.save(model_dir / "registry.json")
    return registry, model_dir


def api_settings(database_url: str, model_dir: Path, **overrides: object) -> ApiSettings:
    values: dict[str, object] = {
        "api_keys": API_KEY,
        "database_url": database_url,
        "model_dir": model_dir,
        "model_cache_dir": model_dir / "cache",
        "model_registry": model_dir / "registry.json",
        "rate_limit_per_minute": 1000,
    }
    values.update(overrides)
    return ApiSettings(**values)  # type: ignore[arg-type]


@pytest.fixture
def api_database(tmp_path: Path) -> str:
    url = f"sqlite:///{(tmp_path / 'api.db').as_posix()}"
    upgrade_database(url)
    return url


@pytest.fixture
def client(tiny_models: tuple[ModelRegistry, Path], api_database: str) -> Iterator[TestClient]:
    registry, model_dir = tiny_models
    settings = api_settings(api_database, model_dir)
    inspector = FabricInspector.from_settings(settings)
    app = create_app(settings, inspector)
    engine = build_engine(api_database)
    with session_scope(build_session_factory(engine)) as session:
        seed_demo(session, registry, settings, frames_per_roll=20)
    engine.dispose()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth() -> dict[str, str]:
    return {"X-API-Key": API_KEY}
