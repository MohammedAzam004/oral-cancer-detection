from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from utils.predictor import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LABEL_MODE,
    DEFAULT_PREPROCESS_MODE,
    DEFAULT_THRESHOLD,
    DEFAULT_UNCERTAINTY_MARGIN,
    predict_image,
)


SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

POSITIVE_DIR_NAMES = {
    "1",
    "cancer",
    "lesion",
    "malignant",
    "oralcancer",
    "oscc",
    "positive",
    "tumor",
    "tumour",
}

NEGATIVE_DIR_NAMES = {
    "0",
    "benign",
    "control",
    "healthy",
    "negative",
    "nocancer",
    "noncancer",
    "normal",
}


@dataclass(frozen=True)
class PerformanceReport:
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    specificity: float | None = None
    f1_score: float | None = None
    balanced_accuracy: float | None = None
    threshold: float | None = None
    total_samples: int | None = None
    skipped_samples: int = 0
    tp: int | None = None
    tn: int | None = None
    fp: int | None = None
    fn: int | None = None
    source: str = "unknown"
    generated_at: str | None = None
    dataset_path: str | None = None
    notes: str | None = None

    @classmethod
    def from_confusion_matrix(
        cls,
        tp: int,
        tn: int,
        fp: int,
        fn: int,
        skipped_samples: int = 0,
        source: str = "local_dataset",
        dataset_path: str | None = None,
        notes: str | None = None,
        threshold: float | None = None,
    ) -> "PerformanceReport":
        total = tp + tn + fp + fn
        if total <= 0:
            raise ValueError("No evaluated samples were available to compute metrics.")

        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1_score = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        balanced_accuracy = (recall + specificity) / 2

        return cls(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            specificity=specificity,
            f1_score=f1_score,
            balanced_accuracy=balanced_accuracy,
            threshold=threshold,
            total_samples=total,
            skipped_samples=skipped_samples,
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
            source=source,
            dataset_path=dataset_path,
            notes=notes,
        )


def _normalize_dir_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def infer_is_cancer_dir(directory_name: str) -> bool | None:
    normalized = _normalize_dir_name(directory_name)
    if normalized in POSITIVE_DIR_NAMES:
        return True
    if normalized in NEGATIVE_DIR_NAMES:
        return False
    return None


def has_labeled_dataset_structure(dataset_dir: str | Path) -> bool:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists() or not dataset_path.is_dir():
        return False

    labels = {
        infer_is_cancer_dir(child.name)
        for child in dataset_path.iterdir()
        if child.is_dir()
    }
    return True in labels and False in labels


def collect_labeled_images(dataset_dir: str | Path) -> list[tuple[Path, bool]]:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")
    if not dataset_path.is_dir():
        raise ValueError(f"Dataset path is not a directory: {dataset_path}")

    labeled_images: list[tuple[Path, bool]] = []
    for child in sorted(dataset_path.iterdir()):
        if not child.is_dir():
            continue

        label = infer_is_cancer_dir(child.name)
        if label is None:
            continue

        for image_path in sorted(child.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                labeled_images.append((image_path, label))

    labels = {label for _, label in labeled_images}
    if not labeled_images or not ({True, False} <= labels):
        raise ValueError(
            "Evaluation dataset must contain labeled image folders such as `cancer/` and `non_cancer/`."
        )

    return labeled_images


def evaluate_dataset(
    model,
    dataset_dir: str | Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    threshold: float = DEFAULT_THRESHOLD,
    preprocess_mode: str = DEFAULT_PREPROCESS_MODE,
    label_mode: str = DEFAULT_LABEL_MODE,
    uncertainty_margin: float = DEFAULT_UNCERTAINTY_MARGIN,
) -> PerformanceReport:
    tp = tn = fp = fn = skipped = 0

    for image_path, expected_cancer in collect_labeled_images(dataset_dir):
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            skipped += 1
            continue

        prediction = predict_image(
            model=model,
            img_bgr=img_bgr,
            image_size=image_size,
            threshold=threshold,
            preprocess_mode=preprocess_mode,
            label_mode=label_mode,
            uncertainty_margin=uncertainty_margin,
        )

        if prediction.is_cancer and expected_cancer:
            tp += 1
        elif prediction.is_cancer and not expected_cancer:
            fp += 1
        elif not prediction.is_cancer and expected_cancer:
            fn += 1
        else:
            tn += 1

    return PerformanceReport.from_confusion_matrix(
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        skipped_samples=skipped,
        source="local_dataset",
        dataset_path=str(Path(dataset_dir).resolve()),
        threshold=threshold,
    )


def _read_float(payload: dict, *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return float(value)
    return None


def _read_int(payload: dict, *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return int(value)
    return None


def load_metrics_report(report_path: str | Path) -> PerformanceReport:
    report_file = Path(report_path)
    payload = json.loads(report_file.read_text(encoding="utf-8"))
    source = str(payload.get("source", "metrics_report"))
    notes = payload.get("notes")
    dataset_path = payload.get("dataset_path")
    metrics = payload.get("metrics")

    if metrics is None and "bundled_model" in payload:
        metrics = payload.get("bundled_model", {}).get("test_metrics", {})
        source = f"{source}: bundled_model"
        if dataset_path is None and payload.get("dataset_roots"):
            dataset_path = ", ".join(str(path) for path in payload["dataset_roots"])
        if notes is None:
            notes_parts: list[str] = []
            preprocess = payload.get("preprocess")
            if preprocess:
                notes_parts.append(
                    f"Validated preprocessing: {str(preprocess).replace('_', ' ')}."
                )
            label_semantics = payload.get("label_semantics")
            if label_semantics:
                notes_parts.append(f"Label semantics: {label_semantics}.")
            bundled_accuracy = _read_float(metrics, "accuracy")
            finetuned_accuracy = _read_float(
                payload.get("finetuned_model", {}).get("test_metrics", {}),
                "accuracy",
            )
            if (
                bundled_accuracy is not None
                and finetuned_accuracy is not None
                and bundled_accuracy >= finetuned_accuracy
            ):
                notes_parts.append(
                    "The bundled model outperformed the fine-tuned checkpoint on the holdout split, so the bundled model remains the serving model."
                )
            notes = " ".join(notes_parts) or None

    if metrics is None:
        metrics = payload

    recall = _read_float(metrics, "recall", "sensitivity")
    specificity = _read_float(metrics, "specificity")
    balanced_accuracy = _read_float(metrics, "balanced_accuracy")
    if balanced_accuracy is None and recall is not None and specificity is not None:
        balanced_accuracy = (recall + specificity) / 2

    report = PerformanceReport(
        accuracy=_read_float(metrics, "accuracy"),
        precision=_read_float(metrics, "precision"),
        recall=recall,
        specificity=specificity,
        f1_score=_read_float(metrics, "f1_score", "f1"),
        balanced_accuracy=balanced_accuracy,
        threshold=_read_float(metrics, "threshold"),
        total_samples=_read_int(metrics, "total_samples", "sample_count"),
        skipped_samples=_read_int(metrics, "skipped_samples") or 0,
        tp=_read_int(metrics, "tp", "true_positive"),
        tn=_read_int(metrics, "tn", "true_negative"),
        fp=_read_int(metrics, "fp", "false_positive"),
        fn=_read_int(metrics, "fn", "false_negative"),
        source=source,
        generated_at=payload.get("generated_at"),
        dataset_path=dataset_path,
        notes=notes,
    )

    if report.accuracy is None:
        raise ValueError("Metrics report must include an `accuracy` field.")

    return report
