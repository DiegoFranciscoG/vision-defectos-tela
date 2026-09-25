"""One command from raw data to served models: `fabric-inspection reproduce`.

download -> manifest + split (leakage check) -> baseline classifier -> PatchCore per backbone ->
evaluation on the shared test split -> ONNX export + INT8 -> registry.json + reports/metrics.json.

Every decision (epoch, backbone, threshold, INT8 or not) is taken on validation. The test split is
only read to report the final numbers.
"""

import json
import logging
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score

from fabric_inspection import __version__
from fabric_inspection.data.download import download_dataset
from fabric_inspection.data.manifest import (
    assign_splits,
    find_leakage,
    scan_dataset,
    split_summary,
    write_manifest,
)
from fabric_inspection.domain.catalog import CLASS_CODES
from fabric_inspection.ml import figures
from fabric_inspection.ml.backbones import BACKBONES
from fabric_inspection.ml.classifier import (
    ClassifierConfig,
    defect_scores,
    predict_probabilities,
    train_classifier,
)
from fabric_inspection.ml.datasets import Sample, load_samples, stack
from fabric_inspection.ml.export import (
    export_onnx,
    max_abs_difference,
    onnx_session,
    quantize_int8,
    run_onnx,
)
from fabric_inspection.ml.metrics import (
    best_pixel_threshold,
    binary_report,
    choose_thresholds,
    multiclass_report,
    pixel_auroc,
)
from fabric_inspection.ml.patchcore import PatchCore, PatchCoreConfig, fit_patchcore, score_samples
from fabric_inspection.ml.reproducibility import set_seed
from fabric_inspection.ml.tracking import code_version, log_artifacts, log_metrics, tracked_run
from fabric_inspection.registry import ExperimentInfo, ModelEntry, ModelRegistry, sha256_file

logger = logging.getLogger(__name__)

WEIGHTS_LICENSE = "CC-BY-NC-SA-4.0"
RELEASE_URL = "https://github.com/DiegoFranciscoG/vision-defectos-tela/releases/download/v1.0.0"
PARITY_TOLERANCE = 1e-3
# INT8 is served only if validation AUPRC drops less than this (decided on val, never on test).
INT8_MAX_AUPRC_DROP = 0.01
# Backbones whose validation AUPRC is this close are considered tied; the lighter one wins.
BACKBONE_TIE = 0.005


@dataclass
class PipelineConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    manifest_path: Path = Path("data/manifest/carpet.csv")
    weights_dir: Path | None = Path(".cache/weights")
    model_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    registry_path: Path = Path("models/registry.json")
    tracking_uri: str = "sqlite:///mlflow.db"
    image_size: int = 256
    seed: int = 42
    patchcore_backbones: tuple[str, ...] = ("resnet18", "wide_resnet50_2")
    coreset_ratio: float = 0.01
    threshold_policy: str = "max_f1_val"
    bundle_version: str = __version__
    release_url: str = RELEASE_URL
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    run_official_protocol: bool = True
    dataset_dir: Path | None = None  # tests point this to a synthetic dataset


def _split(samples: list[Sample], name: str) -> list[Sample]:
    return [sample for sample in samples if sample.record.split == name]


def _targets(samples: list[Sample]) -> np.ndarray:
    return np.array([sample.is_defect for sample in samples])


def _codes(samples: list[Sample]) -> list[str]:
    return [sample.record.class_code for sample in samples]


def _upsampled_cam(cam: np.ndarray, size: int) -> np.ndarray:
    positive = np.maximum(cam, 0)
    peak = positive.max()
    scaled = (positive / peak) if peak > 0 else positive
    image = Image.fromarray(scaled.astype(np.float32))
    return np.asarray(image.resize((size, size), Image.Resampling.BILINEAR))


def _onnx_scores(session: Any, samples: list[Sample], batch: int = 8) -> list[np.ndarray]:
    """Run an ONNX model over samples and concatenate every output."""
    chunks: list[list[np.ndarray]] = []
    for start in range(0, len(samples), batch):
        chunks.append(run_onnx(session, stack(samples[start : start + batch]).numpy()))
    return [np.concatenate([chunk[i] for chunk in chunks]) for i in range(len(chunks[0]))]


def _latency_ms(session: Any, samples: list[Sample], repeats: int = 30) -> dict[str, float]:
    """Single-image CPU latency (batch 1, 1 thread) after 5 warm-up runs."""
    inputs = [stack([sample]).numpy() for sample in samples[: max(1, min(len(samples), 10))]]
    for item in inputs[:5]:
        run_onnx(session, item)
    timings = []
    for index in range(repeats):
        start = time.perf_counter()
        run_onnx(session, inputs[index % len(inputs)])
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "p50": round(statistics.median(timings), 1),
        "p95": round(float(np.percentile(timings, 95)), 1),
    }


def _environment() -> dict[str, str]:
    packages = ("torch", "timm", "onnxruntime", "scikit-learn", "numpy")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        **{name: version(name) for name in packages},
    }


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:  # noqa: PLR0915 - linear script
    started = time.perf_counter()
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = config.reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = config.model_dir
    onnx_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data: download (pinned SHA-256), manifest, split and leakage check.
    dataset_dir = config.dataset_dir or download_dataset(config.raw_dir)
    raw_root = dataset_dir.parents[1] if config.dataset_dir is None else dataset_dir.parent
    records = assign_splits(scan_dataset(dataset_dir, raw_root), config.seed)
    leakage = find_leakage(records)
    if leakage:
        raise RuntimeError(f"Data leakage between splits: {leakage[:5]}")
    manifest_sha = write_manifest(records, config.manifest_path)
    logger.info("Manifest %s: %s", manifest_sha[:12], split_summary(records))
    samples = load_samples(
        records, raw_root, config.processed_dir / str(config.image_size), config.image_size
    )
    train, val, test = (_split(samples, name) for name in ("train", "val", "test"))
    y_val, y_test = _targets(val), _targets(test)
    commit = code_version()
    common = {
        "seed": config.seed,
        "image_size": config.image_size,
        "data_manifest_sha256": manifest_sha,
        "code_version": commit,
    }
    metrics: dict[str, Any] = {
        "dataset": {
            "name": "MVTec AD carpet",
            "license": "CC-BY-NC-SA-4.0",
            "manifest_sha256": manifest_sha,
            "splits": split_summary(records),
            "leakage_pairs": 0,
        },
        "protocol": "stratified per image: good 60/20/20, defects 40/20/40 (train/val/test)",
        "threshold_policy": config.threshold_policy,
        "models": {},
    }

    # 2. Baseline: supervised classifier (uses train defects; test is never seen).
    generator = set_seed(config.seed)
    classifier_params = {**common, "model": "classifier", **asdict(config.classifier)}
    classifier_params.pop("history")
    with tracked_run(config.tracking_uri, "baseline-classifier", classifier_params) as run_id:
        classifier = train_classifier(
            train, val, config.classifier, weights_dir=config.weights_dir, generator=generator
        )
        val_probs = predict_probabilities(classifier, val)
        test_probs = predict_probabilities(classifier, test)
        thresholds = choose_thresholds(y_val, defect_scores(val_probs))
        threshold = thresholds.get(config.threshold_policy)
        report = binary_report(
            y_test, defect_scores(test_probs), _codes(test), threshold, seed=config.seed
        )
        multiclass = multiclass_report(
            np.array([sample.label for sample in test]),
            test_probs.argmax(axis=1),
            list(CLASS_CODES),
        )
        classifier_metrics = {
            "val_auprc": float(average_precision_score(y_val, defect_scores(val_probs))),
            "thresholds_val": asdict(thresholds),
            "test": report.as_dict(),
            "test_multiclass": multiclass,
            "history": config.classifier.history,
        }
        log_metrics({key: classifier_metrics[key] for key in ("val_auprc", "test")})
        classifier_run = run_id
    metrics["models"]["classifier_resnet18"] = classifier_metrics

    # 3. PatchCore per backbone (fit on train good images only).
    detectors: dict[str, tuple[PatchCore, dict[str, Any], str]] = {}
    test_curves = {"classifier (baseline)": (y_test, defect_scores(test_probs))}
    for backbone in config.patchcore_backbones:
        set_seed(config.seed)
        pc_config = PatchCoreConfig(
            backbone=backbone, coreset_ratio=config.coreset_ratio, seed=config.seed
        )
        params = {**common, "model": "patchcore", **asdict(pc_config)}
        with tracked_run(config.tracking_uri, f"patchcore-{backbone}", params) as run_id:
            normal_train = [sample for sample in train if not sample.is_defect]
            model = fit_patchcore(
                normal_train, config.image_size, pc_config, weights_dir=config.weights_dir
            )
            val_scores, val_maps = score_samples(model, val)
            test_scores, test_maps = score_samples(model, test)
            thresholds = choose_thresholds(y_val, val_scores)
            threshold = thresholds.get(config.threshold_policy)
            val_masks = np.stack([sample.mask for sample in val])
            test_masks = np.stack([sample.mask for sample in test])
            pc_metrics = {
                "val_auprc": float(average_precision_score(y_val, val_scores)),
                "thresholds_val": asdict(thresholds),
                "pixel_threshold_val": best_pixel_threshold(val_masks, val_maps),
                "test": binary_report(
                    y_test, test_scores, _codes(test), threshold, seed=config.seed
                ).as_dict(),
                "test_pixel_auroc": pixel_auroc(test_masks, test_maps),
                "memory_bank": list(model.memory_bank.shape),
                "parameters": sum(p.numel() for p in model.parameters()),
            }
            log_metrics({k: pc_metrics[k] for k in ("val_auprc", "test", "test_pixel_auroc")})
        metrics["models"][f"patchcore_{backbone}"] = pc_metrics
        detectors[backbone] = (model, pc_metrics, run_id)
        test_curves[f"PatchCore {backbone}"] = (y_test, test_scores)

    # 4. Backbone selection on validation (pre-registered rule, see BACKBONE_TIE).
    best_val = max(item[1]["val_auprc"] for item in detectors.values())
    tied = [
        name for name, item in detectors.items() if item[1]["val_auprc"] >= best_val - BACKBONE_TIE
    ]
    selected = min(tied, key=lambda name: detectors[name][1]["parameters"])
    detector, detector_metrics, detector_run = detectors[selected]
    metrics["selected_backbone"] = {
        "backbone": selected,
        "rule": f"best val AUPRC; ties within {BACKBONE_TIE} go to the model with fewer parameters",
    }

    # 5. Sanity check against the literature (official MVTec split, PatchCore only).
    if config.run_official_protocol:
        set_seed(config.seed)
        official_train = [s for s in samples if s.record.source_split == "train"]
        official_test = [s for s in samples if s.record.source_split == "test"]
        official = fit_patchcore(
            official_train,
            config.image_size,
            PatchCoreConfig(
                backbone=selected, coreset_ratio=config.coreset_ratio, seed=config.seed
            ),
            weights_dir=config.weights_dir,
        )
        scores, maps = score_samples(official, official_test)
        y_official = _targets(official_test)
        metrics["mvtec_official_protocol"] = {
            "backbone": selected,
            "image_auroc": binary_report(
                y_official,
                scores,
                _codes(official_test),
                float(np.median(scores)),
                seed=config.seed,
            ).auroc,
            "image_auprc": float(average_precision_score(y_official, scores)),
            "pixel_auroc": pixel_auroc(np.stack([s.mask for s in official_test]), maps),
        }
        del official

    # 6. Export to ONNX, check parity, quantize and decide INT8 on validation.
    sample_batch = stack(val[:2])
    detector_fp32 = export_onnx(
        detector, sample_batch[:1], onnx_dir / "patchcore.fp32.onnx", ["score", "anomaly_map"]
    )
    classifier_fp32 = export_onnx(
        classifier, sample_batch[:1], onnx_dir / "classifier.fp32.onnx", ["probabilities", "cam"]
    )
    parity = {
        "patchcore": max_abs_difference(detector, onnx_session(detector_fp32), sample_batch),
        "classifier": max_abs_difference(classifier, onnx_session(classifier_fp32), sample_batch),
    }
    if max(parity.values()) > PARITY_TOLERANCE:
        raise RuntimeError(f"ONNX output differs from PyTorch: {parity}")
    calibration = [stack([sample]).numpy() for sample in train if not sample.is_defect][:32]
    detector_int8 = quantize_int8(detector_fp32, onnx_dir / "patchcore.int8.onnx", calibration)
    classifier_int8 = quantize_int8(classifier_fp32, onnx_dir / "classifier.int8.onnx", calibration)

    serving: dict[str, Any] = {"onnx_parity_max_abs_diff": parity}
    chosen: dict[str, tuple[Path, bool]] = {}
    for name, fp32, int8 in (
        ("patchcore", detector_fp32, detector_int8),
        ("classifier", classifier_fp32, classifier_int8),
    ):
        variants: dict[str, dict[str, Any]] = {}
        for label, path in (("fp32", fp32), ("int8", int8)):
            session = onnx_session(path)
            val_out = _onnx_scores(session, val)
            val_score = val_out[0] if name == "patchcore" else defect_scores(val_out[0])
            variants[label] = {
                "val_auprc": float(average_precision_score(y_val, val_score)),
                "size_mb": round(path.stat().st_size / 1e6, 2),
                "latency_ms": _latency_ms(session, test),
            }
        drop = variants["fp32"]["val_auprc"] - variants["int8"]["val_auprc"]
        use_int8 = drop <= INT8_MAX_AUPRC_DROP
        chosen[name] = (int8 if use_int8 else fp32, use_int8)
        serving[name] = {
            **variants,
            "served": "int8" if use_int8 else "fp32",
            "val_auprc_drop": drop,
        }

    # 7. Thresholds of the served ONNX models, recomputed on validation, and test report.
    detector_path, detector_quantized = chosen["patchcore"]
    classifier_path, classifier_quantized = chosen["classifier"]
    detector_session = onnx_session(detector_path)
    val_scores, val_maps = _onnx_scores(detector_session, val)
    test_scores, test_maps = _onnx_scores(detector_session, test)
    served_thresholds = choose_thresholds(y_val, val_scores)
    served_threshold = served_thresholds.get(config.threshold_policy)
    pixel_threshold = best_pixel_threshold(np.stack([s.mask for s in val]), val_maps)
    served_report = binary_report(
        y_test, test_scores, _codes(test), served_threshold, seed=config.seed
    )
    classifier_session = onnx_session(classifier_path)
    test_probs_onnx, test_cams = _onnx_scores(classifier_session, test)
    serving["served_detector_test"] = served_report.as_dict()
    serving["served_classifier_test_multiclass"] = multiclass_report(
        np.array([sample.label for sample in test]),
        test_probs_onnx.argmax(axis=1),
        list(CLASS_CODES),
    )
    serving["thresholds_val"] = asdict(served_thresholds)
    serving["pixel_threshold_val"] = pixel_threshold
    metrics["serving"] = serving

    # 8. Figures.
    figures.pr_curves(test_curves, figures_dir / "pr_curves.png")
    figures.confusion(
        multiclass["confusion"],
        list(CLASS_CODES),
        "Baseline classifier (test)",
        figures_dir / "confusion_classifier.png",
    )
    figures.confusion(
        served_report.confusion,
        ["good", "defect"],
        f"PatchCore {selected} served (test)",
        figures_dir / "confusion_detector.png",
    )
    figures.score_histogram(
        test_scores, y_test, served_threshold, figures_dir / "score_distribution.png"
    )
    example_index = []
    for code in CLASS_CODES[1:]:
        example_index.extend([i for i, s in enumerate(test) if s.record.class_code == code][:1])
    predicted = test_probs_onnx.argmax(axis=1)
    figures.examples(
        [test[i] for i in example_index],
        test_maps[example_index],
        np.stack(
            [_upsampled_cam(test_cams[i, predicted[i]], config.image_size) for i in example_index]
        ),
        pixel_threshold,
        figures_dir / "examples.png",
    )

    # 9. Registry (what the API serves) and metrics report.
    reference = [round(float(s), 5) for s, y in zip(val_scores, y_val, strict=True) if y == 0]

    def entry(
        path: Path, quantized: bool, name: str, info: ExperimentInfo, **extra: Any
    ) -> ModelEntry:
        return ModelEntry(
            name=name,
            version=config.bundle_version,
            file=path.name,
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
            quantized=quantized,
            experiment=info,
            **extra,
        )

    registry = ModelRegistry(
        bundle_version=config.bundle_version,
        release_url=config.release_url,
        weights_license=WEIGHTS_LICENSE,
        image_size=config.image_size,
        detector=entry(
            detector_path,
            detector_quantized,
            f"patchcore-{selected}",
            ExperimentInfo(
                mlflow_run_id=detector_run,
                model_type="patchcore",
                backbone=selected,
                protocol="stratified",
                params={"coreset_ratio": config.coreset_ratio, "image_size": config.image_size},
                metrics={
                    "test": served_report.as_dict(),
                    "val_auprc": detector_metrics["val_auprc"],
                },
                seed=config.seed,
                data_manifest_sha256=manifest_sha,
                code_version=commit,
            ),
            threshold=served_threshold,
            threshold_policy=config.threshold_policy,
            pixel_threshold=pixel_threshold,
            reference_scores=reference,
        ),
        classifier=entry(
            classifier_path,
            classifier_quantized,
            "defect-classifier-resnet18",
            ExperimentInfo(
                mlflow_run_id=classifier_run,
                model_type="classifier",
                backbone="resnet18",
                protocol="stratified",
                params={"epochs": config.classifier.epochs, "image_size": config.image_size},
                metrics={"test_macro_f1": serving["served_classifier_test_multiclass"]["macro_f1"]},
                seed=config.seed,
                data_manifest_sha256=manifest_sha,
                code_version=commit,
            ),
            class_codes=list(CLASS_CODES),
        ),
    )
    registry.save(config.registry_path)
    metrics["environment"] = _environment()
    metrics["backbone_weights_license"] = {name: spec.license for name, spec in BACKBONES.items()}
    metrics["runtime_minutes"] = round((time.perf_counter() - started) / 60, 1)
    metrics_path = config.reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with tracked_run(config.tracking_uri, "served-bundle", {**common, "backbone": selected}):
        log_metrics({"served": serving["served_detector_test"]})
        log_artifacts([metrics_path, config.registry_path, *sorted(figures_dir.glob("*.png"))])
    return metrics
