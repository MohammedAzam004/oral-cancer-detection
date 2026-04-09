from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.evaluation import (
    PerformanceReport,
    has_labeled_dataset_structure,
    infer_is_cancer_dir,
    load_metrics_report,
)


class EvaluationTests(unittest.TestCase):
    def test_infer_is_cancer_dir_understands_common_folder_names(self) -> None:
        self.assertTrue(infer_is_cancer_dir("cancer"))
        self.assertTrue(infer_is_cancer_dir("OSCC"))
        self.assertFalse(infer_is_cancer_dir("non_cancer"))
        self.assertFalse(infer_is_cancer_dir("healthy"))
        self.assertIsNone(infer_is_cancer_dir("misc"))

    def test_performance_report_from_confusion_matrix(self) -> None:
        report = PerformanceReport.from_confusion_matrix(tp=8, tn=10, fp=2, fn=4)

        self.assertAlmostEqual(report.accuracy, 18 / 24)
        self.assertAlmostEqual(report.precision, 8 / 10)
        self.assertAlmostEqual(report.recall, 8 / 12)
        self.assertAlmostEqual(report.specificity, 10 / 12)
        self.assertAlmostEqual(report.f1_score, 0.7272727272)

    def test_has_labeled_dataset_structure_requires_positive_and_negative_dirs(self) -> None:
        root = Path("evaluation")
        children = [Path("cancer"), Path("non_cancer")]
        with patch.object(Path, "exists", return_value=True), patch.object(
            Path,
            "is_dir",
            autospec=True,
            side_effect=lambda _self: True,
        ), patch.object(Path, "iterdir", return_value=iter(children)):
            self.assertTrue(has_labeled_dataset_structure(root))

    def test_load_metrics_report_reads_json_metrics(self) -> None:
        payload = json.dumps(
            {
                "source": "validation_run",
                "generated_at": "2026-04-08",
                "notes": "Held-out validation split.",
                "metrics": {
                    "accuracy": 0.91,
                    "precision": 0.88,
                    "recall": 0.86,
                    "specificity": 0.94,
                    "f1_score": 0.87,
                    "total_samples": 240,
                    "tp": 86,
                    "tn": 132,
                    "fp": 8,
                    "fn": 14,
                },
            }
        )
        with patch.object(Path, "read_text", return_value=payload):
            report = load_metrics_report(Path("metrics.json"))

        self.assertEqual(report.source, "validation_run")
        self.assertEqual(report.generated_at, "2026-04-08")
        self.assertAlmostEqual(report.accuracy, 0.91)
        self.assertEqual(report.total_samples, 240)
        self.assertEqual(report.tp, 86)
        self.assertEqual(report.fn, 14)


if __name__ == "__main__":
    unittest.main()
