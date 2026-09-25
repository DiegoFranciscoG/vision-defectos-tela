"""Report figures (PNG). They show MVTec AD images, so they carry its CC BY-NC-SA 4.0 license."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.metrics import precision_recall_curve

from fabric_inspection.domain.regions import extract_regions
from fabric_inspection.ml.datasets import Sample


def pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, scores) in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax.step(recall, precision, where="post", label=name)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test)", ylim=(0, 1.02))
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def confusion(matrix: list[list[int]], labels: list[str], title: str, path: Path) -> None:
    data = np.array(matrix)
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2, 1.1 * len(labels) + 1))
    ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for (row, col), value in np.ndenumerate(data):
        color = "white" if value > data.max() / 2 else "black"
        ax.text(col, row, str(value), ha="center", va="center", color=color)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def score_histogram(
    scores: np.ndarray, is_defect: np.ndarray, threshold: float, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(scores.min(), scores.max(), 30).tolist()
    ax.hist(scores[is_defect == 0], bins=bins, alpha=0.7, label="good")
    ax.hist(scores[is_defect == 1], bins=bins, alpha=0.7, label="defect")
    ax.axvline(threshold, color="black", linestyle="--", label="threshold (val)")
    ax.set(xlabel="Anomaly score", ylabel="Images", title="Detector scores on test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def examples(
    samples: list[Sample],
    maps: np.ndarray,
    cams: np.ndarray,
    pixel_threshold: float,
    path: Path,
) -> None:
    """One row per example: image with ground truth, PatchCore map with boxes, classifier CAM."""
    fig, axes = plt.subplots(len(samples), 3, figsize=(9, 3 * len(samples)))
    axes = np.atleast_2d(axes)
    for row, (sample, anomaly, cam) in enumerate(zip(samples, maps, cams, strict=True)):
        image = np.asarray(sample.image)
        axes[row, 0].imshow(image)
        axes[row, 0].contour(sample.mask, levels=[0.5], colors="lime", linewidths=1)
        axes[row, 0].set_title(f"{sample.record.class_code} (ground truth)")
        axes[row, 1].imshow(image)
        axes[row, 1].imshow(anomaly, cmap="jet", alpha=0.45)
        size = image.shape[1], image.shape[0]
        for region in extract_regions(anomaly, pixel_threshold, original_size=size):
            axes[row, 1].add_patch(
                Rectangle(
                    (region.x, region.y), region.width, region.height, fill=False, color="white"
                )
            )
        axes[row, 1].set_title("PatchCore map + regions")
        axes[row, 2].imshow(image)
        axes[row, 2].imshow(cam, cmap="jet", alpha=0.45)
        axes[row, 2].set_title("Classifier CAM (= Grad-CAM)")
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
