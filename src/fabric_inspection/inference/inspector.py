"""Serving-time inspection with ONNX Runtime (CPU). No PyTorch at runtime.

detector (PatchCore): is there a defect, where, and how long is it?
classifier (ResNet-18 + CAM): which type (a hole counts 4 points) and why (CAM).
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from fabric_inspection.config import Settings
from fabric_inspection.domain.catalog import GOOD, get_class
from fabric_inspection.domain.four_point import points_for_defect
from fabric_inspection.domain.regions import extract_regions
from fabric_inspection.preprocessing import preprocess
from fabric_inspection.registry import ModelEntry, ModelRegistry, resolve_model_file


@dataclass(frozen=True, slots=True)
class DetectedRegion:
    class_code: str
    x: int
    y: int
    width: int
    height: int
    length_mm: float
    is_hole: bool
    points: int
    peak_score: float


@dataclass(frozen=True, slots=True)
class InspectionResult:
    score: float
    threshold: float
    is_defective: bool
    predicted_class: str | None
    class_confidence: float | None
    class_probabilities: dict[str, float]
    regions: list[DetectedRegion]
    anomaly_map: np.ndarray  # (S, S) float32, model resolution
    cam: np.ndarray  # (h, w) float32, activation map of the reported class
    latency_ms: float
    width: int
    height: int


def _session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # No memory arena: lower resident memory on small instances (Render free has 512 MB).
    options.enable_cpu_mem_arena = False
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class FabricInspector:
    def __init__(
        self,
        registry: ModelRegistry,
        detector_path: Path,
        classifier_path: Path,
        *,
        mm_per_pixel: float,
        threads: int = 1,
    ) -> None:
        if registry.detector.threshold is None:
            raise ValueError("The detector has no threshold in the registry")
        if not registry.classifier.class_codes:
            raise ValueError("The classifier has no class list in the registry")
        self.registry = registry
        self.threshold = registry.detector.threshold
        self.pixel_threshold = min(
            registry.detector.pixel_threshold or registry.detector.threshold, self.threshold
        )
        self.class_codes = registry.classifier.class_codes
        self.image_size = registry.image_size
        self.mm_per_pixel = mm_per_pixel
        self._detector = _session(detector_path, threads)
        self._classifier = _session(classifier_path, threads)

    @classmethod
    def from_settings(cls, settings: Settings) -> "FabricInspector":
        registry = ModelRegistry.load(settings.model_registry)
        base_url = settings.model_base_url or registry.release_url

        def resolve(entry: ModelEntry) -> Path:
            return resolve_model_file(
                entry,
                model_dir=settings.model_dir,
                cache_dir=settings.model_cache_dir,
                base_url=base_url,
            )

        return cls(
            registry,
            resolve(registry.detector),
            resolve(registry.classifier),
            mm_per_pixel=settings.mm_per_pixel,
        )

    def inspect(self, image: Image.Image) -> InspectionResult:
        started = time.perf_counter()
        width, height = image.size
        batch = preprocess(image, self.image_size)
        score, anomaly = self._detector.run(None, {"images": batch})
        probabilities, cams = self._classifier.run(None, {"images": batch})
        image_score = float(score[0])
        anomaly_map = anomaly[0].astype(np.float32)
        probs = probabilities[0]
        is_defective = image_score >= self.threshold

        predicted_class: str | None = None
        confidence: float | None = None
        regions: list[DetectedRegion] = []
        good_index = self.class_codes.index(GOOD)
        cam_index = int(np.argmax(probs))
        if is_defective:
            # The detector decides *whether*; the classifier only names the type, so the most
            # likely defect class is reported even if the classifier alone would say "good".
            defect_indices = [i for i in range(len(self.class_codes)) if i != good_index]
            cam_index = max(defect_indices, key=lambda i: probs[i])
            predicted_class = self.class_codes[cam_index]
            confidence = float(probs[cam_index])
            defect = get_class(predicted_class)
            for region in extract_regions(
                anomaly_map, self.pixel_threshold, original_size=(width, height)
            ):
                length = max(region.length_mm(self.mm_per_pixel), 0.1)
                regions.append(
                    DetectedRegion(
                        class_code=predicted_class,
                        x=region.x,
                        y=region.y,
                        width=region.width,
                        height=region.height,
                        length_mm=length,
                        is_hole=defect.is_hole,
                        points=points_for_defect(length, is_hole=defect.is_hole),
                        peak_score=region.peak_score,
                    )
                )
        return InspectionResult(
            score=image_score,
            threshold=self.threshold,
            is_defective=is_defective,
            predicted_class=predicted_class,
            class_confidence=confidence,
            class_probabilities={
                code: round(float(p), 4) for code, p in zip(self.class_codes, probs, strict=True)
            },
            regions=regions,
            anomaly_map=anomaly_map,
            cam=cams[0, cam_index].astype(np.float32),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            width=width,
            height=height,
        )
