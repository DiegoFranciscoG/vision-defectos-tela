"""Image preprocessing shared by training and serving, so both see exactly the same pixels.

Only NumPy and Pillow: the inference API does not depend on PyTorch.
"""

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_for_model(image: Image.Image, size: int) -> Image.Image:
    """RGB and square resize without cropping, so defects near the border are kept (S13)."""
    rgb = image.convert("RGB")
    if rgb.size == (size, size):
        return rgb
    return rgb.resize((size, size), Image.Resampling.BILINEAR)


def resize_mask(mask: Image.Image, size: int) -> np.ndarray:
    """Binary mask resized with nearest neighbour (labels must not be interpolated)."""
    resized = mask.convert("L").resize((size, size), Image.Resampling.NEAREST)
    return (np.asarray(resized) > 127).astype(np.uint8)


def normalize(image: Image.Image) -> np.ndarray:
    """HWC uint8 -> CHW float32 normalised with ImageNet statistics (not dataset statistics)."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def preprocess(image: Image.Image, size: int) -> np.ndarray:
    """Model input batch of one image: shape (1, 3, size, size)."""
    return normalize(resize_for_model(image, size))[np.newaxis]
