"""Turn a pixel anomaly map into defect regions with a physical length."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True, slots=True)
class Region:
    """Axis-aligned box in the coordinates of the original image."""

    x: int
    y: int
    width: int
    height: int
    area_px: int
    peak_score: float

    def length_mm(self, mm_per_pixel: float) -> float:
        """Defect length = longest side of its bounding box (S1 gives the scale)."""
        return round(max(self.width, self.height) * mm_per_pixel, 1)


def extract_regions(
    anomaly_map: np.ndarray,
    threshold: float,
    *,
    original_size: tuple[int, int],
    min_area_px: int = 4,
    max_regions: int = 10,
) -> list[Region]:
    """Connected components of `anomaly_map >= threshold`, scaled to `original_size` (w, h).

    Regions are sorted by peak score and very small ones (noise) are dropped.
    """
    if anomaly_map.ndim != 2:
        raise ValueError("anomaly_map must be a 2-D array")
    map_h, map_w = anomaly_map.shape
    orig_w, orig_h = original_size
    scale_x, scale_y = orig_w / map_w, orig_h / map_h
    labels, count = ndimage.label(anomaly_map >= threshold)
    regions: list[Region] = []
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        rows, cols = box
        component = labels[box] == index
        area = int(component.sum())
        if area < min_area_px:
            continue
        peak = float(anomaly_map[box][component].max())
        regions.append(
            Region(
                x=int(cols.start * scale_x),
                y=int(rows.start * scale_y),
                width=max(1, round((cols.stop - cols.start) * scale_x)),
                height=max(1, round((rows.stop - rows.start) * scale_y)),
                area_px=round(area * scale_x * scale_y),
                peak_score=peak,
            )
        )
    if count and not regions:
        # Every component was tiny: keep the one holding the global maximum.
        return extract_regions(
            anomaly_map, threshold, original_size=original_size, min_area_px=1, max_regions=1
        )
    regions.sort(key=lambda region: region.peak_score, reverse=True)
    return regions[:max_regions]
