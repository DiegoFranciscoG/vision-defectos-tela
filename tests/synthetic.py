"""Tiny synthetic dataset with the MVTec AD folder layout (no licensed images in the repo)."""

from pathlib import Path

import numpy as np
from PIL import Image

DEFECTS = ("color", "cut", "hole", "metal_contamination", "thread")


def _texture(rng: np.random.Generator, size: int) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    phase = rng.uniform(0, 2 * np.pi)
    weave = 0.5 + 0.25 * np.sin(x * 0.9 + phase) * np.sin(y * 0.9 + phase)
    noise = rng.normal(0, 0.08, (size, size))
    gray = np.clip(weave + noise, 0, 1)
    return np.stack([gray * 0.8, gray * 0.6, gray * 0.5], axis=-1)


def make_image(
    rng: np.random.Generator, size: int, defect: str | None
) -> tuple[Image.Image, Image.Image | None]:
    image = _texture(rng, size)
    mask = None
    if defect is not None:
        mask_array = np.zeros((size, size), dtype=np.uint8)
        x0, y0 = rng.integers(size // 8, size // 2, 2)
        if defect in {"cut", "thread"}:
            mask_array[y0 : y0 + 2, x0 : x0 + size // 3] = 255
        else:
            mask_array[y0 : y0 + size // 6, x0 : x0 + size // 6] = 255
        color = {"color": (0.9, 0.1, 0.1), "hole": (0.0, 0.0, 0.0)}.get(defect, (1.0, 1.0, 1.0))
        image[mask_array > 0] = color
        mask = Image.fromarray(mask_array)
    return Image.fromarray((image * 255).astype(np.uint8)), mask


def make_dataset(
    root: Path,
    *,
    size: int = 64,
    good_train: int = 16,
    good_test: int = 4,
    per_defect: int = 5,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    carpet = root / "carpet"
    for split, count in (("train", good_train), ("test", good_test)):
        folder = carpet / split / "good"
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            make_image(rng, size, None)[0].save(folder / f"{index:03d}.png")
    for defect in DEFECTS:
        folder = carpet / "test" / defect
        masks = carpet / "ground_truth" / defect
        folder.mkdir(parents=True, exist_ok=True)
        masks.mkdir(parents=True, exist_ok=True)
        for index in range(per_defect):
            image, mask = make_image(rng, size, defect)
            image.save(folder / f"{index:03d}.png")
            assert mask is not None
            mask.save(masks / f"{index:03d}_mask.png")
    return carpet
