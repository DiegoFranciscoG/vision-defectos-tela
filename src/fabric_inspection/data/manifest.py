"""Dataset manifest: one row per image with hashes, label and split (rules R2 and R10).

The split is assigned per image, before any resizing or augmentation, stratified by class and
with a fixed seed. Exact duplicates (SHA-256) and near duplicates (dHash) across splits are
reported as leakage.
"""

import csv
import hashlib
import io
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from fabric_inspection.domain.catalog import CLASS_CODES, GOOD
from fabric_inspection.registry import sha256_file

SPLITS = ("train", "val", "test")
# Protocol B fractions (train, val, test): good 60/20/20, defects 40/20/40.
GOOD_FRACTIONS = (0.6, 0.2)
DEFECT_FRACTIONS = (0.4, 0.2)
NEAR_DUPLICATE_HAMMING = 4


@dataclass(frozen=True, slots=True)
class ImageRecord:
    relative_path: str
    class_code: str
    source_split: str
    sha256: str
    dhash: str
    width: int
    height: int
    mask_path: str
    split: str = ""


def dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Difference hash: 64 bits comparing adjacent pixels of a 9x8 grayscale thumbnail."""
    small = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = int("".join("1" if bit else "0" for bit in bits), 2)
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def scan_dataset(dataset_dir: Path, root: Path) -> list[ImageRecord]:
    """Walk the MVTec folder layout: {train,test}/<class>/*.png and ground_truth masks."""
    records: list[ImageRecord] = []
    for source_split in ("train", "test"):
        split_dir = dataset_dir / source_split
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            class_code = class_dir.name
            if class_code not in CLASS_CODES:
                raise ValueError(f"Unknown class folder: {class_dir}")
            for image_path in sorted(class_dir.glob("*.png")):
                mask = ""
                if class_code != GOOD:
                    mask_file = (
                        dataset_dir / "ground_truth" / class_code / f"{image_path.stem}_mask.png"
                    )
                    if not mask_file.is_file():
                        raise FileNotFoundError(f"Missing mask for {image_path}")
                    mask = mask_file.relative_to(root).as_posix()
                with Image.open(image_path) as image:
                    width, height = image.size
                    image_dhash = dhash(image)
                records.append(
                    ImageRecord(
                        relative_path=image_path.relative_to(root).as_posix(),
                        class_code=class_code,
                        source_split=source_split,
                        sha256=sha256_file(image_path),
                        dhash=image_dhash,
                        width=width,
                        height=height,
                        mask_path=mask,
                    )
                )
    return records


def _split_counts(total: int, fractions: tuple[float, float]) -> tuple[int, int, int]:
    n_train = round(total * fractions[0])
    n_val = round(total * fractions[1])
    return n_train, n_val, total - n_train - n_val


def assign_splits(records: list[ImageRecord], seed: int) -> list[ImageRecord]:
    """Stratified per-image split. Records are ordered by SHA-256 first, so the result depends
    only on the file contents and the seed, not on the file system order."""
    by_class: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_class[record.class_code].append(record)
    rng = np.random.default_rng(seed)
    assigned: list[ImageRecord] = []
    for class_code in sorted(by_class):
        items = sorted(by_class[class_code], key=lambda record: record.sha256)
        order = rng.permutation(len(items))
        fractions = GOOD_FRACTIONS if class_code == GOOD else DEFECT_FRACTIONS
        n_train, n_val, _ = _split_counts(len(items), fractions)
        for position, index in enumerate(order):
            split = (
                "train" if position < n_train else "val" if position < n_train + n_val else "test"
            )
            assigned.append(replace(items[index], split=split))
    return sorted(assigned, key=lambda record: record.relative_path)


def find_leakage(
    records: list[ImageRecord], max_hamming: int = NEAR_DUPLICATE_HAMMING
) -> list[tuple[str, str, str]]:
    """Pairs of images in different splits that are exact or near duplicates."""
    issues: list[tuple[str, str, str]] = []
    seen: dict[str, ImageRecord] = {}
    for record in records:
        other = seen.get(record.sha256)
        if other is not None and other.split != record.split:
            issues.append((other.relative_path, record.relative_path, "sha256"))
        seen.setdefault(record.sha256, record)
    for first, second in combinations(records, 2):
        if first.split != second.split and hamming(first.dhash, second.dhash) <= max_hamming:
            issues.append((first.relative_path, second.relative_path, "dhash"))
    return issues


FIELDS = tuple(ImageRecord.__dataclass_fields__)


def write_manifest(records: list[ImageRecord], path: Path) -> str:
    """Write the CSV with LF line endings and return its SHA-256 (the data version)."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(asdict(record))
    content = buffer.getvalue().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def read_manifest(path: Path) -> list[ImageRecord]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            ImageRecord(
                relative_path=row["relative_path"],
                class_code=row["class_code"],
                source_split=row["source_split"],
                sha256=row["sha256"],
                dhash=row["dhash"],
                width=int(row["width"]),
                height=int(row["height"]),
                mask_path=row["mask_path"],
                split=row["split"],
            )
            for row in csv.DictReader(handle)
        ]


def split_summary(records: list[ImageRecord]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {split: defaultdict(int) for split in SPLITS}
    for record in records:
        summary[record.split][record.class_code] += 1
    return {split: dict(sorted(counts.items())) for split, counts in summary.items()}
