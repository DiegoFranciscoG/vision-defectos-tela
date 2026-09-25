"""Keep the database in sync with models/registry.json (which ONNX files are in production)."""

from dataclasses import dataclass

from sqlalchemy.orm import Session

from fabric_inspection.db.models import Experiment, ModelVersion
from fabric_inspection.exceptions import BusinessRuleError
from fabric_inspection.registry import ModelEntry, ModelRegistry
from fabric_inspection.repository.repositories import ModelVersionRepository


@dataclass(frozen=True, slots=True)
class ServedModels:
    detector: ModelVersion
    classifier: ModelVersion


def _register_entry(
    repository: ModelVersionRepository, entry: ModelEntry, registry: ModelRegistry
) -> ModelVersion:
    existing = repository.get_by_sha256(entry.sha256)
    if existing is not None:
        if existing.status != "production":
            repository.archive_production(existing.name)
            existing.status = "production"
        return existing
    if repository.get_by_name_version(entry.name, entry.version) is not None:
        raise BusinessRuleError(
            f"{entry.name} {entry.version} is already registered with another SHA-256; "
            "publish new weights under a new version"
        )
    info = entry.experiment
    experiment = repository.get_experiment(info.mlflow_run_id)
    if experiment is None:
        experiment = Experiment(
            mlflow_run_id=info.mlflow_run_id,
            model_type=info.model_type,
            backbone=info.backbone,
            protocol=info.protocol,
            params=info.params,
            metrics=info.metrics,
            seed=info.seed,
            data_manifest_sha256=info.data_manifest_sha256,
            code_version=info.code_version,
        )
        repository.add(experiment)
    repository.archive_production(entry.name)
    version = ModelVersion(
        experiment_id=experiment.id,
        name=entry.name,
        version=entry.version,
        onnx_sha256=entry.sha256,
        onnx_size_bytes=entry.size_bytes,
        quantized=entry.quantized,
        threshold=entry.threshold,
        threshold_policy=entry.threshold_policy,
        pixel_threshold=entry.pixel_threshold,
        reference_scores=entry.reference_scores,
        weights_license=registry.weights_license,
        release_url=f"{registry.release_url.rstrip('/')}/{entry.file}",
        status="production",
    )
    repository.add(version)
    return version


def register_models(session: Session, registry: ModelRegistry) -> ServedModels:
    repository = ModelVersionRepository(session)
    return ServedModels(
        detector=_register_entry(repository, registry.detector, registry),
        classifier=_register_entry(repository, registry.classifier, registry),
    )
