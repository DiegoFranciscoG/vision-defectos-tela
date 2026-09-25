"""Heat-map overlays as PNG with Pillow and NumPy only (the API image has no matplotlib)."""

import base64
import io

import numpy as np
from PIL import Image, ImageDraw

# Five-stop blue -> cyan -> green -> yellow -> red palette.
_STOPS = np.array(
    [[0, 0, 128], [0, 160, 255], [0, 200, 80], [255, 220, 0], [220, 0, 0]], dtype=np.float32
)


def colorize(values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to RGB uint8."""
    scaled = np.clip(values, 0, 1) * (len(_STOPS) - 1)
    low = np.floor(scaled).astype(int)
    high = np.minimum(low + 1, len(_STOPS) - 1)
    fraction = (scaled - low)[..., None]
    rgb = _STOPS[low] * (1 - fraction) + _STOPS[high] * fraction
    colored: np.ndarray = rgb.astype(np.uint8)
    return colored


def overlay(
    image: Image.Image,
    heat: np.ndarray,
    *,
    low: float,
    high: float,
    boxes: list[tuple[int, int, int, int]] | None = None,
    alpha: float = 0.45,
    max_side: int = 512,
) -> Image.Image:
    """Blend a heat map (any resolution) over the image; values are scaled from low to high."""
    base = image.convert("RGB")
    scale = min(1.0, max_side / max(base.size))
    size = (max(1, round(base.width * scale)), max(1, round(base.height * scale)))
    base = base.resize(size, Image.Resampling.BILINEAR)
    span = max(high - low, 1e-6)
    heat_image = Image.fromarray(np.asarray(heat, dtype=np.float32)).resize(
        size, Image.Resampling.BILINEAR
    )
    colored = Image.fromarray(colorize((np.asarray(heat_image) - low) / span))
    blended = Image.blend(base, colored, alpha)
    if boxes:
        draw = ImageDraw.Draw(blended)
        for x, y, width, height in boxes:
            draw.rectangle(
                [x * scale, y * scale, (x + width) * scale, (y + height) * scale],
                outline=(255, 255, 255),
                width=2,
            )
    return blended


def to_png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
