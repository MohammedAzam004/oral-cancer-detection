from __future__ import annotations

import unittest

import numpy as np

from utils.predictor import (
    DEFAULT_PREPROCESS_MODE,
    PredictionResult,
    get_model_input_size,
    interpret_model_score,
    normalize_preprocess_mode,
    preprocess_image,
)


class DummyModel:
    input_shape = (None, 300, 300, 3)


class PredictorTests(unittest.TestCase):
    def test_interpret_model_score_marks_cancer_below_threshold(self) -> None:
        result = interpret_model_score(0.20)

        self.assertIsInstance(result, PredictionResult)
        self.assertTrue(result.is_cancer)
        self.assertFalse(result.is_uncertain)
        self.assertEqual(result.label, "Cancer detected")
        self.assertAlmostEqual(result.cancer_probability, 0.80)
        self.assertAlmostEqual(result.non_cancer_probability, 0.20)
        self.assertAlmostEqual(result.confidence, 0.80)
        self.assertEqual(result.focus_class, "cancer")

    def test_interpret_model_score_marks_no_cancer_above_threshold(self) -> None:
        result = interpret_model_score(0.82)

        self.assertFalse(result.is_cancer)
        self.assertFalse(result.is_uncertain)
        self.assertEqual(result.label, "No cancer detected")
        self.assertAlmostEqual(result.cancer_probability, 0.18)
        self.assertAlmostEqual(result.non_cancer_probability, 0.82)
        self.assertAlmostEqual(result.confidence, 0.82)
        self.assertEqual(result.focus_class, "no_cancer")

    def test_interpret_model_score_can_flip_label_mapping(self) -> None:
        result = interpret_model_score(0.82, label_mode="output_is_cancer")

        self.assertTrue(result.is_cancer)
        self.assertAlmostEqual(result.cancer_probability, 0.82)
        self.assertAlmostEqual(result.non_cancer_probability, 0.18)

    def test_interpret_model_score_marks_borderline_values_for_review(self) -> None:
        result = interpret_model_score(0.52, uncertainty_margin=0.05)

        self.assertTrue(result.is_uncertain)
        self.assertEqual(result.label, "Needs review")
        self.assertEqual(result.interpretation, "needs_review")

    def test_preprocess_image_defaults_to_legacy_normalized_mode(self) -> None:
        img = np.full((24, 24, 3), 255, dtype=np.uint8)

        processed = preprocess_image(img, image_size=12)

        self.assertEqual(processed.shape, (1, 12, 12, 3))
        self.assertEqual(processed.dtype, np.float32)
        self.assertEqual(DEFAULT_PREPROCESS_MODE, "legacy_normalized")
        self.assertAlmostEqual(float(processed.max()), 1.0)

    def test_preprocess_image_can_use_raw_pixel_mode(self) -> None:
        img = np.full((24, 24, 3), 255, dtype=np.uint8)

        processed = preprocess_image(img, image_size=12, preprocess_mode="raw_pixels")

        self.assertAlmostEqual(float(processed.max()), 255.0)

    def test_normalize_preprocess_mode_falls_back_to_default(self) -> None:
        self.assertEqual(normalize_preprocess_mode("unexpected"), DEFAULT_PREPROCESS_MODE)

    def test_get_model_input_size_reads_model_shape(self) -> None:
        self.assertEqual(get_model_input_size(DummyModel()), 300)


if __name__ == "__main__":
    unittest.main()
