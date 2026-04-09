from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf

from utils.predictor import (
    DEFAULT_PREPROCESS_MODE,
    DEFAULT_THRESHOLD,
    preprocess_image,
)


def _find_last_feature_layer(model) -> str:
    for layer in reversed(model.layers):
        output = getattr(layer, "output", None)
        shape = getattr(output, "shape", None)
        if shape is not None and len(shape) == 4:
            return layer.name

    raise ValueError("Unable to locate a convolutional feature layer for Grad-CAM.")


def _resolve_focus_class(prediction_score: tf.Tensor, focus_on: str, threshold: float) -> str:
    if focus_on in {"cancer", "no_cancer"}:
        return focus_on

    score = float(prediction_score.numpy())
    return "cancer" if score <= threshold else "no_cancer"


def generate_gradcam(
    model,
    img_bgr: np.ndarray,
    focus_on: str = "predicted",
    image_size: int = 300,
    threshold: float = DEFAULT_THRESHOLD,
    preprocess_mode: str = DEFAULT_PREPROCESS_MODE,
) -> np.ndarray:
    """Generate a Grad-CAM overlay for the requested class direction."""

    img_tensor = preprocess_image(
        img_bgr,
        image_size=image_size,
        preprocess_mode=preprocess_mode,
    )
    last_feature_layer = _find_last_feature_layer(model)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_feature_layer).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model([img_tensor], training=False)
        prediction = tf.reshape(prediction, [-1])[0]
        target_class = _resolve_focus_class(prediction, focus_on, threshold)
        loss = 1.0 - prediction if target_class == "cancer" else prediction

    grads = tape.gradient(loss, conv_output)
    if grads is None:
        raise ValueError("Gradients could not be computed for the selected class.")

    grads = grads[0]
    conv_output = conv_output[0]
    weights = tf.reduce_mean(grads, axis=(0, 1)).numpy()

    cam = np.zeros(conv_output.shape[:2], dtype=np.float32)
    for index, weight in enumerate(weights):
        cam += weight * conv_output[:, :, index]

    cam = np.maximum(cam, 0)
    max_value = float(cam.max())
    if max_value > 0:
        cam /= max_value

    cam = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 0.62, heatmap, 0.38, 0)
