from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Precision, Recall
from tensorflow.keras.optimizers import Adam


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_MODEL_PATH = Path("models/EfficientNetV2S_mouth_cancer.keras")
DEFAULT_OUTPUT_MODEL_PATH = Path("artifacts/EfficientNetV2S_mouth_cancer_finetuned.keras")
DEFAULT_REPORT_DIR = Path("reports")
DEFAULT_SEED = 42
DEFAULT_IMAGE_SIZE = 300


@dataclass(frozen=True)
class ManifestItem:
    image_hash: str
    is_cancer: bool
    path: str
    source_root: str

    @property
    def target(self) -> float:
        # The existing bundled model's sigmoid output behaves like a no-cancer score.
        return 0.0 if self.is_cancer else 1.0

    @property
    def label_name(self) -> str:
        return "cancer" if self.is_cancer else "non_cancer"


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    balanced_accuracy: float
    threshold: float
    total_samples: int
    tp: int
    tn: int
    fp: int
    fn: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the oral cancer model.")
    parser.add_argument(
        "--dataset-root",
        action="append",
        dest="dataset_roots",
        help="Dataset root with cancer/non-cancer subfolders. Can be passed multiple times.",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the existing Keras model to fine-tune.",
    )
    parser.add_argument(
        "--output-model-path",
        default=str(DEFAULT_OUTPUT_MODEL_PATH),
        help="Path to save the fine-tuned model.",
    )
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="Directory where metrics and logs should be written.",
    )
    parser.add_argument("--epochs", type=int, default=4, help="Maximum fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def infer_is_cancer_from_dir(directory_name: str) -> bool:
    normalized = "".join(character for character in directory_name.lower() if character.isalnum())
    if "non" in normalized or normalized in {"healthy", "normal", "negative"}:
        return False
    return True


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_manifest(dataset_roots: list[Path]) -> tuple[list[ManifestItem], list[str]]:
    by_hash: dict[str, list[ManifestItem]] = defaultdict(list)
    conflicts: list[str] = []

    for root in dataset_roots:
        if not root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")

        for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            is_cancer = infer_is_cancer_from_dir(class_dir.name)
            for image_path in class_dir.rglob("*"):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue

                image_hash = md5_file(image_path)
                item = ManifestItem(
                    image_hash=image_hash,
                    is_cancer=is_cancer,
                    path=str(image_path),
                    source_root=str(root),
                )
                by_hash[image_hash].append(item)

    manifest: list[ManifestItem] = []
    for image_hash, items in by_hash.items():
        labels = {item.is_cancer for item in items}
        if len(labels) > 1:
            conflicts.append(image_hash)
            continue

        chosen = min(items, key=lambda item: (len(item.path), item.path))
        manifest.append(chosen)

    manifest.sort(key=lambda item: (item.label_name, item.path))
    return manifest, conflicts


def filter_readable_items(manifest: list[ManifestItem]) -> tuple[list[ManifestItem], list[str]]:
    readable: list[ManifestItem] = []
    unreadable_paths: list[str] = []
    for item in manifest:
        image = cv2.imread(item.path, cv2.IMREAD_COLOR)
        if image is None:
            unreadable_paths.append(item.path)
            continue
        readable.append(item)
    return readable, unreadable_paths


def stratified_split(
    manifest: list[ManifestItem],
    seed: int,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, list[ManifestItem]]:
    rng = random.Random(seed)
    buckets = {True: [], False: []}
    for item in manifest:
        buckets[item.is_cancer].append(item)

    splits = {"train": [], "val": [], "test": []}
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n_total = len(bucket)
        n_test = max(1, round(n_total * test_ratio))
        n_val = max(1, round(n_total * val_ratio))
        splits["test"].extend(bucket[:n_test])
        splits["val"].extend(bucket[n_test : n_test + n_val])
        splits["train"].extend(bucket[n_test + n_val :])

    for split_items in splits.values():
        split_items.sort(key=lambda item: item.path)
    return splits


class ImageSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        items: list[ManifestItem],
        batch_size: int,
        image_size: int,
        training: bool = False,
        seed: int = DEFAULT_SEED,
    ) -> None:
        super().__init__()
        self.items = list(items)
        self.batch_size = batch_size
        self.image_size = image_size
        self.training = training
        self.rng = np.random.default_rng(seed)
        self.indexes = np.arange(len(self.items))
        if self.training:
            self.rng.shuffle(self.indexes)

    def __len__(self) -> int:
        return math.ceil(len(self.items) / self.batch_size)

    def on_epoch_end(self) -> None:
        if self.training:
            self.rng.shuffle(self.indexes)

    def _augment(self, image: np.ndarray) -> np.ndarray:
        if self.rng.random() < 0.5:
            image = cv2.flip(image, 1)

        angle = float(self.rng.uniform(-12.0, 12.0))
        scale = float(self.rng.uniform(0.95, 1.05))
        matrix = cv2.getRotationMatrix2D((self.image_size / 2, self.image_size / 2), angle, scale)
        image = cv2.warpAffine(
            image,
            matrix,
            (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        brightness = float(self.rng.uniform(0.92, 1.08))
        image = np.clip(image * brightness, 0.0, 1.0)
        return image

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_items = [self.items[i] for i in batch_indexes]

        batch_x = np.empty((len(batch_items), self.image_size, self.image_size, 3), dtype=np.float32)
        batch_y = np.empty((len(batch_items),), dtype=np.float32)

        for row, item in enumerate(batch_items):
            image = cv2.imread(item.path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image: {item.path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.image_size, self.image_size)).astype("float32") / 255.0
            if self.training:
                image = self._augment(image)

            batch_x[row] = image
            batch_y[row] = item.target

        return batch_x, batch_y


def predict_scores(
    model,
    items: list[ManifestItem],
    image_size: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    sequence = ImageSequence(items, batch_size=batch_size, image_size=image_size, training=False)
    scores = model.predict(sequence, verbose=0).reshape(-1)
    labels = np.array([item.target for item in items], dtype=np.float32)
    return scores, labels


def metrics_from_scores(scores: np.ndarray, labels: np.ndarray, threshold: float) -> BinaryMetrics:
    predictions = (scores <= threshold).astype(np.int32)
    cancer_labels = (labels == 0.0).astype(np.int32)

    tp = int(np.sum((predictions == 1) & (cancer_labels == 1)))
    tn = int(np.sum((predictions == 0) & (cancer_labels == 0)))
    fp = int(np.sum((predictions == 1) & (cancer_labels == 0)))
    fn = int(np.sum((predictions == 0) & (cancer_labels == 1)))
    total = tp + tn + fp + fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    balanced_accuracy = (recall + specificity) / 2

    return BinaryMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1_score=f1_score,
        balanced_accuracy=balanced_accuracy,
        threshold=threshold,
        total_samples=total,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )


def find_best_threshold(scores: np.ndarray, labels: np.ndarray) -> BinaryMetrics:
    best: BinaryMetrics | None = None
    for raw_threshold in range(1, 100):
        threshold = raw_threshold / 100
        candidate = metrics_from_scores(scores, labels, threshold)
        if best is None:
            best = candidate
            continue

        candidate_key = (
            candidate.balanced_accuracy,
            candidate.f1_score,
            candidate.accuracy,
            -abs(candidate.threshold - 0.5),
        )
        best_key = (
            best.balanced_accuracy,
            best.f1_score,
            best.accuracy,
            -abs(best.threshold - 0.5),
        )
        if candidate_key > best_key:
            best = candidate

    assert best is not None
    return best


def evaluate_model(
    model,
    splits: dict[str, list[ManifestItem]],
    image_size: int,
    batch_size: int,
) -> dict[str, object]:
    val_scores, val_labels = predict_scores(model, splits["val"], image_size=image_size, batch_size=batch_size)
    test_scores, test_labels = predict_scores(model, splits["test"], image_size=image_size, batch_size=batch_size)
    best_val = find_best_threshold(val_scores, val_labels)
    test_metrics = metrics_from_scores(test_scores, test_labels, best_val.threshold)
    return {
        "validation_threshold_search": asdict(best_val),
        "test_metrics": asdict(test_metrics),
    }


def class_weights(train_items: list[ManifestItem]) -> dict[int, float]:
    positives = sum(1 for item in train_items if item.target == 1.0)
    negatives = len(train_items) - positives
    total = len(train_items)
    return {
        0: total / (2 * negatives),
        1: total / (2 * positives),
    }


def image_size_from_model(model) -> int:
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if input_shape and len(input_shape) >= 3 and input_shape[1]:
        return int(input_shape[1])
    return DEFAULT_IMAGE_SIZE


def train_finetuned_model(
    model_path: Path,
    output_model_path: Path,
    report_dir: Path,
    splits: dict[str, list[ManifestItem]],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    model = load_model(str(model_path), compile=False)
    image_size = image_size_from_model(model)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=BinaryCrossentropy(),
        metrics=[
            BinaryAccuracy(name="accuracy"),
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc"),
        ],
    )

    train_sequence = ImageSequence(
        splits["train"],
        batch_size=batch_size,
        image_size=image_size,
        training=True,
        seed=seed,
    )
    val_sequence = ImageSequence(
        splits["val"],
        batch_size=batch_size,
        image_size=image_size,
        training=False,
        seed=seed,
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = report_dir / "best_finetuned.weights.h5"
    callbacks = [
        EarlyStopping(monitor="val_auc", mode="max", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=1, min_lr=1e-6),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
        CSVLogger(str(report_dir / "finetune_history.csv")),
    ]

    history = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=epochs,
        class_weight=class_weights(splits["train"]),
        callbacks=callbacks,
        verbose=2,
    )

    if checkpoint_path.exists():
        model.load_weights(str(checkpoint_path))

    model.save(str(output_model_path))
    model = load_model(str(output_model_path), compile=False)

    evaluation = evaluate_model(
        model,
        splits=splits,
        image_size=image_size,
        batch_size=batch_size,
    )
    evaluation["history"] = history.history
    return evaluation


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_roots = (
        [Path(root) for root in args.dataset_roots]
        if args.dataset_roots
        else [
            Path("Oral Cancer/Oral Cancer Dataset"),
            Path("Oral cancer Dataset 2.0/OC Dataset kaggle new"),
        ]
    )

    manifest, conflicts = collect_manifest(dataset_roots)
    manifest, unreadable_paths = filter_readable_items(manifest)
    splits = stratified_split(manifest, seed=args.seed)

    model_path = Path(args.model_path)
    output_model_path = Path(args.output_model_path)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    bundled_model = load_model(str(model_path), compile=False)
    image_size = image_size_from_model(bundled_model)
    bundled_metrics = evaluate_model(
        bundled_model,
        splits=splits,
        image_size=image_size,
        batch_size=args.batch_size,
    )

    finetuned_metrics = train_finetuned_model(
        model_path=model_path,
        output_model_path=output_model_path,
        report_dir=report_dir,
        splits=splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    report = {
        "source": "local_dataset_holdout_training",
        "seed": args.seed,
        "dataset_roots": [str(path) for path in dataset_roots],
        "clean_unique_images": len(manifest),
        "conflicting_duplicate_hashes_excluded": len(conflicts),
        "unreadable_images_excluded": len(unreadable_paths),
        "split_counts": {
            split_name: {
                "total": len(split_items),
                "cancer": sum(1 for item in split_items if item.is_cancer),
                "non_cancer": sum(1 for item in split_items if not item.is_cancer),
            }
            for split_name, split_items in splits.items()
        },
        "label_semantics": "sigmoid output is interpreted as no-cancer score",
        "preprocess": "legacy_normalized_rgb",
        "bundled_model": bundled_metrics,
        "finetuned_model": finetuned_metrics,
        "output_model_path": str(output_model_path),
    }

    manifest_path = report_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "items": [asdict(item) for item in manifest],
                "conflicts": conflicts,
                "unreadable_paths": unreadable_paths,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path = report_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["bundled_model"]["test_metrics"], indent=2))
    print(json.dumps(report["finetuned_model"]["test_metrics"], indent=2))
    print(f"Report saved to: {report_path}")
    print(f"Fine-tuned model saved to: {output_model_path}")


if __name__ == "__main__":
    main()
