from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.orm import Session, sessionmaker

from fabric_inspection.config import QualitySettings
from fabric_inspection.db.migrate import upgrade_database
from fabric_inspection.db.session import build_engine, build_session_factory
from fabric_inspection.registry import ExperimentInfo, ModelEntry, ModelRegistry


def _experiment(model_type: str, run_id: str) -> ExperimentInfo:
    return ExperimentInfo(
        mlflow_run_id=run_id,
        model_type=model_type,  # type: ignore[arg-type]
        backbone="resnet18",
        protocol="stratified",
        params={"image_size": 64},
        metrics={"test_auprc": 0.9},
        seed=42,
        data_manifest_sha256="a" * 64,
        code_version="test",
    )


def make_registry(
    detector_sha: str = "1" * 64,
    classifier_sha: str = "2" * 64,
    *,
    detector_version: str = "0.0.1",
    detector_file: str = "detector.onnx",
    classifier_file: str = "classifier.onnx",
    image_size: int = 64,
) -> ModelRegistry:
    reference = np.random.default_rng(0).normal(1.0, 0.1, 120).round(4).tolist()
    return ModelRegistry(
        bundle_version="0.0.1",
        release_url="https://example.invalid/releases/v0.0.1",
        weights_license="CC-BY-NC-SA-4.0",
        image_size=image_size,
        detector=ModelEntry(
            name="patchcore-test",
            version=detector_version,
            file=detector_file,
            sha256=detector_sha,
            size_bytes=10,
            quantized=False,
            threshold=1.4,
            threshold_policy="max_f1_val",
            pixel_threshold=1.2,
            reference_scores=reference,
            experiment=_experiment("patchcore", "run-detector"),
        ),
        classifier=ModelEntry(
            name="classifier-test",
            version="0.0.1",
            file=classifier_file,
            sha256=classifier_sha,
            size_bytes=10,
            quantized=False,
            class_codes=["good", "color", "cut", "hole", "metal_contamination", "thread"],
            experiment=_experiment("classifier", "run-classifier"),
        ),
    )


@pytest.fixture
def registry() -> ModelRegistry:
    return make_registry()


@pytest.fixture
def quality_settings() -> QualitySettings:
    return QualitySettings(mm_per_pixel=0.1, max_points_per_100_sq_yd=40, warning_ratio=0.8)


@pytest.fixture
def database_url(tmp_path: Path) -> str:
    url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    upgrade_database(url)
    return url


@pytest.fixture
def session_factory(database_url: str) -> Iterator[sessionmaker[Session]]:
    engine = build_engine(database_url)
    yield build_session_factory(engine)
    engine.dispose()


@pytest.fixture
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as db_session:
        yield db_session
