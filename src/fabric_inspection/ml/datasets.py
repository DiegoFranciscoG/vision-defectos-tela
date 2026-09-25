"""Cached, resized images and the PyTorch dataset used for training and evaluation."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from fabric_inspection.data.manifest import ImageRecord
from fabric_inspection.domain.catalog import CLASS_CODES, GOOD
from fabric_inspection.preprocessing import normalize, resize_for_model, resize_mask


@dataclass(frozen=True, slots=True)
class Sample:
    record: ImageRecord
    image: Image.Image  # RGB, already resized to the model size
    mask: np.ndarray  # (size, size) uint8, zeros for good images

    @property
    def label(self) -> int:
        return CLASS_CODES.index(self.record.class_code)

    @property
    def is_defect(self) -> int:
        return int(self.record.class_code != GOOD)


def load_samples(
    records: list[ImageRecord], raw_root: Path, cache_dir: Path, size: int
) -> list[Sample]:
    """Resize once and cache as PNG (lossless), so every epoch reads the same pixels."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for record in records:
        image_cache = cache_dir / f"{record.sha256}.png"
        mask_cache = cache_dir / f"{record.sha256}_mask.png"
        if not image_cache.is_file():
            with Image.open(raw_root / record.relative_path) as original:
                resize_for_model(original, size).save(image_cache)
        if record.mask_path and not mask_cache.is_file():
            with Image.open(raw_root / record.mask_path) as original_mask:
                Image.fromarray(resize_mask(original_mask, size) * 255).save(mask_cache)
        with Image.open(image_cache) as cached:
            image: Image.Image = cached.convert("RGB")
        if record.mask_path:
            with Image.open(mask_cache) as cached_mask:
                mask: np.ndarray = (np.asarray(cached_mask) > 127).astype(np.uint8)
        else:
            mask = np.zeros((size, size), dtype=np.uint8)
        samples.append(Sample(record=record, image=image, mask=mask))
    return samples


class FabricDataset(Dataset[tuple[torch.Tensor, int]]):
    """Normalised tensors; training adds flips and 90-degree rotations (textures are isotropic)."""

    def __init__(self, samples: list[Sample], *, augment: bool, seed: int = 0) -> None:
        self._arrays = [normalize(sample.image) for sample in samples]
        self._labels = [sample.label for sample in samples]
        self._augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._arrays)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        array = self._arrays[index]
        if self._augment:
            array = np.rot90(array, k=int(self._rng.integers(4)), axes=(1, 2))
            if self._rng.random() < 0.5:
                array = array[:, :, ::-1]
        return torch.from_numpy(np.ascontiguousarray(array)), self._labels[index]


def stack(samples: list[Sample]) -> torch.Tensor:
    return torch.from_numpy(np.stack([normalize(sample.image) for sample in samples]))
