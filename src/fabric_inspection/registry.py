"""Model registry file (models/registry.json): what is served, its hash, threshold and metrics.

The training pipeline writes it; the API and the seed command read it. Weights live in GitHub
Releases and are only loaded when their SHA-256 matches this file (rule R9).
"""

import hashlib
import json
import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_CHUNK = 1024 * 1024


class ExperimentInfo(BaseModel):
    mlflow_run_id: str
    model_type: Literal["classifier", "patchcore"]
    backbone: str
    protocol: Literal["mvtec_official", "stratified"]
    params: dict[str, Any]
    metrics: dict[str, Any]
    seed: int
    data_manifest_sha256: str
    code_version: str


class ModelEntry(BaseModel):
    name: str
    version: str
    file: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    quantized: bool
    threshold: float | None = None
    threshold_policy: str | None = None
    pixel_threshold: float | None = None
    reference_scores: list[float] | None = None
    class_codes: list[str] | None = None
    experiment: ExperimentInfo


class ModelRegistry(BaseModel):
    bundle_version: str
    release_url: str
    weights_license: str
    image_size: int = Field(gt=0)
    detector: ModelEntry
    classifier: ModelEntry

    @classmethod
    def load(cls, path: Path) -> "ModelRegistry":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False)
        path.write_text(payload + "\n", encoding="utf-8")


class ModelIntegrityError(RuntimeError):
    """The model file does not match the registered SHA-256."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_file(
    entry: ModelEntry,
    *,
    model_dir: Path,
    cache_dir: Path,
    base_url: str | None,
) -> Path:
    """Return a local, verified copy of the model: local dir, then cache, then download."""
    for candidate in (model_dir / entry.file, cache_dir / entry.file):
        if candidate.is_file():
            if sha256_file(candidate) == entry.sha256:
                return candidate
            logger.warning("Ignoring %s: SHA-256 does not match the registry", candidate)
    if not base_url:
        raise ModelIntegrityError(
            f"{entry.file} is not available locally and no download URL is configured"
        )
    if not base_url.startswith("https://"):
        raise ModelIntegrityError("Model downloads must use HTTPS")
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"{base_url.rstrip('/')}/{entry.file}"
    logger.info("Downloading %s", url)
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - https enforced
            shutil.copyfileobj(response, tmp, _CHUNK)
    if sha256_file(tmp_path) != entry.sha256:
        tmp_path.unlink(missing_ok=True)
        raise ModelIntegrityError(f"Downloaded {entry.file} does not match the registered SHA-256")
    target = cache_dir / entry.file
    tmp_path.replace(target)
    return target
