"""Download MVTec AD carpet and verify it against a pinned SHA-256 (rule R1, OWASP ML02)."""

import logging
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from fabric_inspection.registry import sha256_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RemoteFile:
    url: str
    sha256: str
    filename: str


# docs/investigacion.md #2. License: CC BY-NC-SA 4.0 (non-commercial use only).
MVTEC_CARPET = RemoteFile(
    url=(
        "https://www.mydrive.ch/shares/150455/eac7fbce84d93a5094e13f391170eca4/download/"
        "420937484-1629959013/carpet.tar.xz"
    ),
    sha256="d9dd5064515a20bd75cf24d223c570579141abd72388beef892780b74dbaa85d",
    filename="carpet.tar.xz",
)
DATASET_SUBDIR = Path("mvtec_ad") / "carpet"


class ChecksumError(RuntimeError):
    """A downloaded file does not match its pinned SHA-256."""


def fetch(remote: RemoteFile, target_dir: Path) -> Path:
    """Download once; reuse the local copy only if its hash still matches."""
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / remote.filename
    if target.is_file() and sha256_file(target) == remote.sha256:
        logger.info("%s already downloaded and verified", remote.filename)
        return target
    if not remote.url.startswith("https://"):
        raise ChecksumError("Only HTTPS downloads are allowed")
    logger.info("Downloading %s", remote.url)
    with tempfile.NamedTemporaryFile(dir=target_dir, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with urllib.request.urlopen(remote.url, timeout=120) as response:  # noqa: S310 - https
            shutil.copyfileobj(response, tmp, 1024 * 1024)
    actual = sha256_file(tmp_path)
    if actual != remote.sha256:
        tmp_path.unlink(missing_ok=True)
        raise ChecksumError(f"{remote.filename}: expected {remote.sha256}, got {actual}")
    tmp_path.replace(target)
    return target


def download_dataset(raw_dir: Path) -> Path:
    """Return the extracted carpet folder, downloading and extracting it if needed."""
    dataset_dir = raw_dir / DATASET_SUBDIR
    if (dataset_dir / "train" / "good").is_dir():
        return dataset_dir
    archive = fetch(MVTEC_CARPET, raw_dir)
    logger.info("Extracting %s", archive.name)
    with tarfile.open(archive, "r:xz") as tar:
        # The "data" filter rejects absolute paths, links outside the target and device files.
        tar.extractall(raw_dir / "mvtec_ad", filter="data")
    return dataset_dir
