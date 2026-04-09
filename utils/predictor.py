from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


DEFAULT_IMAGE_SIZE = 300
DEFAULT_THRESHOLD = 0.50
DEFAULT_PREPROCESS_MODE = "legacy_normalized"
DEFAULT_LABEL_MODE = "output_is_no_cancer"
DEFAULT_UNCERTAINTY_MARGIN = 0.05


@dataclass(frozen=True)
class PredictionResult:
    label: str
    is_cancer: bool
    is_uncertain: bool
    cancer_probability: float
    non_cancer_probability: float
    confidence: float
    raw_score: float
    threshold: float
    label_mode: str
    uncertainty_margin: float

    @property
    def focus_class(self) -> str:
        return "cancer" if self.is_cancer else "no_cancer"

    @property
    def interpretation(self) -> str:
        if self.is_uncertain:
            return "needs_review"
        return self.focus_class


def clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_preprocess_mode(mode: str | None) -> str:
    if mode in {"legacy_normalized", "raw_pixels"}:
        return mode
    return DEFAULT_PREPROCESS_MODE


def normalize_label_mode(mode: str | None) -> str:
    if mode in {"output_is_no_cancer", "output_is_cancer"}:
        return mode
    return DEFAULT_LABEL_MODE


def get_model_input_size(model, default: int = DEFAULT_IMAGE_SIZE) -> int:
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if input_shape and len(input_shape) >= 3 and input_shape[1] and input_shape[2]:
        return int(input_shape[1])

    return default


def preprocess_image(
    img_bgr: np.ndarray,
    image_size: int = DEFAULT_IMAGE_SIZE,
    preprocess_mode: str = DEFAULT_PREPROCESS_MODE,
) -> np.ndarray:
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Input image is empty or unreadable.")

    if image_size <= 0:
        raise ValueError("Image size must be a positive integer.")

    preprocess_mode = normalize_preprocess_mode(preprocess_mode)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size, image_size)).astype("float32")
    if preprocess_mode == "legacy_normalized":
        img_resized = img_resized / 255.0
    return np.expand_dims(img_resized, axis=0)


def interpret_model_score(
    raw_score: float,
    threshold: float = DEFAULT_THRESHOLD,
    label_mode: str = DEFAULT_LABEL_MODE,
    uncertainty_margin: float = DEFAULT_UNCERTAINTY_MARGIN,
) -> PredictionResult:
    label_mode = normalize_label_mode(label_mode)
    uncertainty_margin = max(0.0, min(0.49, float(uncertainty_margin)))
    raw_probability = clamp_probability(raw_score)

    if label_mode == "output_is_cancer":
        cancer_probability = raw_probability
        non_cancer_probability = 1.0 - raw_probability
        is_cancer = cancer_probability >= threshold
    else:
        non_cancer_probability = raw_probability
        cancer_probability = 1.0 - raw_probability
        is_cancer = non_cancer_probability <= threshold

    confidence = cancer_probability if is_cancer else non_cancer_probability
    is_uncertain = abs(raw_probability - threshold) <= uncertainty_margin
    if is_uncertain:
        label = "Needs review"
    else:
        label = "Cancer detected" if is_cancer else "No cancer detected"

    return PredictionResult(
        label=label,
        is_cancer=is_cancer,
        is_uncertain=is_uncertain,
        cancer_probability=cancer_probability,
        non_cancer_probability=non_cancer_probability,
        confidence=confidence,
        raw_score=float(raw_score),
        threshold=threshold,
        label_mode=label_mode,
        uncertainty_margin=uncertainty_margin,
    )


def predict_image(
    model,
    img_bgr: np.ndarray,
    image_size: int = DEFAULT_IMAGE_SIZE,
    threshold: float = DEFAULT_THRESHOLD,
    preprocess_mode: str = DEFAULT_PREPROCESS_MODE,
    label_mode: str = DEFAULT_LABEL_MODE,
    uncertainty_margin: float = DEFAULT_UNCERTAINTY_MARGIN,
) -> PredictionResult:
    img_tensor = preprocess_image(
        img_bgr,
        image_size=image_size,
        preprocess_mode=preprocess_mode,
    )
    raw_score = float(model.predict(img_tensor, verbose=0)[0][0])
    return interpret_model_score(
        raw_score,
        threshold=threshold,
        label_mode=label_mode,
        uncertainty_margin=uncertainty_margin,
    )
