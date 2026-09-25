"""Evaluation for imbalanced data: AUPRC first, thresholds chosen on validation (#17-#19)."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass(frozen=True, slots=True)
class Thresholds:
    max_f1_val: float
    recall_95_val: float
    p99_normal_val: float

    def get(self, policy: str) -> float:
        value: float = getattr(self, policy)
        return value


def choose_thresholds(y_true: np.ndarray, scores: np.ndarray) -> Thresholds:
    """Three policies computed on validation only (R3)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.argmax(f1[:-1]))  # the last PR point has no threshold
    reaching = np.where(recall[:-1] >= 0.95)[0]
    recall_95 = float(thresholds[reaching[-1]]) if reaching.size else float(thresholds[0])
    normal = scores[y_true == 0]
    return Thresholds(
        max_f1_val=float(thresholds[best]),
        recall_95_val=recall_95,
        p99_normal_val=float(np.quantile(normal, 0.99)) if normal.size else float("nan"),
    )


def bootstrap_auprc_ci(
    y_true: np.ndarray, scores: np.ndarray, *, seed: int, rounds: int = 1000, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap, resampling positives and negatives separately (stratified)."""
    rng = np.random.default_rng(seed)
    positives, negatives = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
    values = []
    for _ in range(rounds):
        index = np.concatenate(
            [
                rng.choice(positives, positives.size, replace=True),
                rng.choice(negatives, negatives.size, replace=True),
            ]
        )
        values.append(average_precision_score(y_true[index], scores[index]))
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


@dataclass(frozen=True, slots=True)
class BinaryReport:
    auprc: float
    auprc_ci95: tuple[float, float]
    auroc: float
    threshold: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    confusion: list[list[int]]  # [[tn, fp], [fn, tp]]
    recall_by_class: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def binary_report(
    y_true: np.ndarray,
    scores: np.ndarray,
    class_codes: list[str],
    threshold: float,
    *,
    seed: int,
) -> BinaryReport:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predicted, average="binary", zero_division=0
    )
    codes = np.array(class_codes)
    recall_by_class = {
        code: float(predicted[codes == code].mean()) for code in sorted(set(class_codes) - {"good"})
    }
    return BinaryReport(
        auprc=float(average_precision_score(y_true, scores)),
        auprc_ci95=bootstrap_auprc_ci(y_true, scores, seed=seed),
        auroc=float(roc_auc_score(y_true, scores)),
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        false_positive_rate=float(fp / max(fp + tn, 1)),
        confusion=[[int(tn), int(fp)], [int(fn), int(tp)]],
        recall_by_class=recall_by_class,
    )


def multiclass_report(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    """Per-class precision/recall/F1 and the confusion matrix of the six-class classifier."""
    index = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=index, zero_division=0
    )
    return {
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "macro_f1": float(f1_score(y_true, y_pred, labels=index, average="macro", zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred, labels=index).tolist(),
        "labels": labels,
    }


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float:
    return float(roc_auc_score(masks.reshape(-1), maps.reshape(-1)))


def best_pixel_threshold(masks: np.ndarray, maps: np.ndarray) -> float:
    """Pixel threshold with the best pixel F1 on validation (localisation of the defect)."""
    precision, recall, thresholds = precision_recall_curve(masks.reshape(-1), maps.reshape(-1))
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.argmax(f1[:-1]))])
