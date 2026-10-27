"""Проверки фактического CSV и безопасного манифеста без обучения моделей."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from contracts import TaskSpec
from submission_audit import audit_submission, write_run_manifest


class SubmissionAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "submission.csv"
        self.test = pd.DataFrame({"id": ["0001", "0002", "0003"], "x": [1., 2., 3.]})
        self.test.to_csv(self.root / "test.csv", index=False)
        self.task = TaskSpec(
            task_path=str(self.root / "task.md"), train_path=str(self.root / "train.csv"),
            test_path=str(self.root / "test.csv"), sample_path=None, dictionary_path=None,
            target="response", problem_type="regression", metric="rmse", id_columns=["id"],
            feature_columns=["x"], categorical_columns=[], output_columns=["id", "response"])

    def tearDown(self):
        self.temp.cleanup()

    def write(self, values, columns=None):
        frame = pd.DataFrame({"id": self.test.id, "response": values})
        if columns is not None:
            frame = frame[columns]
        frame.to_csv(self.path, index=False)
        return frame

    def test_leading_zero_ids_hash_and_unrestricted_finite_regression(self):
        self.write([-1e200, 0., 1e200])
        result = audit_submission(self.task, self.test, self.path)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["columns"], self.task.output_columns)
        self.assertEqual(result["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(result["prediction_summary"], {"min": -1e200, "max": 1e200, "mean": 0.})
        self.assertIn("0001", self.path.read_text(encoding="utf-8"))

    def test_wrong_order_duplicate_ids_and_schema_are_rejected(self):
        original = self.write([1., 2., 3.])
        cases = [original.iloc[::-1], original.assign(id=["0001", "0001", "0003"]),
                 original[["response", "id"]], original.iloc[:2], original.rename(columns={"response": "other"}),
                 original.assign(id=["1", "2", "3"])]
        for frame in cases:
            with self.subTest(frame=frame.to_dict("list")):
                frame.to_csv(self.path, index=False)
                with self.assertRaises(ValueError):
                    audit_submission(self.task, self.test, self.path)

    def test_composite_ids_are_unique_as_a_tuple(self):
        test = self.test.assign(part=["a", "b", "a"], id=["0001", "0001", "0002"])
        task = replace(self.task, id_columns=["id", "part"], output_columns=["id", "part", "response"])
        test[["id", "part"]].assign(response=[1, 2, 3]).to_csv(self.path, index=False)
        self.assertEqual(audit_submission(task, test, self.path)["status"], "ok")

    def test_invalid_labels_and_literal_na_label(self):
        task = replace(self.task, problem_type="classification", metric="macro_f1",
                       prediction_kind="label", classes=["NA", "other"])
        self.write(["NA", "other", "NA"])
        result = audit_submission(task, self.test, self.path)
        self.assertEqual(result["label_distribution"], {"NA": 2, "other": 1})
        self.write(["NA", "invalid", "other"])
        with self.assertRaises(ValueError):
            audit_submission(task, self.test, self.path)

    def test_nan_infinity_and_non_numeric_regression_are_rejected(self):
        for value in ("NaN", "nan", "inf", "-Infinity", "", "bad"):
            with self.subTest(value=value):
                self.write([1., value, 3.])
                with self.assertRaises(ValueError):
                    audit_submission(self.task, self.test, self.path)

    def test_probability_ranges_row_sums_and_finiteness(self):
        task = replace(self.task, problem_type="classification", metric="log_loss",
                       prediction_kind="probabilities", classes=["a", "b"],
                       probability_columns=["b", "a"], probability_indices=[1, 0],
                       output_columns=["id", "b", "a"])
        base = pd.DataFrame({"id": self.test.id, "b": [.2, .3, 1.], "a": [.8, .7, 0.]})
        base.to_csv(self.path, index=False)
        result = audit_submission(task, self.test, self.path)
        self.assertEqual(result["probability_row_sums"]["mean"], 1.)
        for first, second in ((.2, .7), (-.1, 1.1), (1.1, -.1), ("NaN", 1.), ("inf", 0.)):
            changed = base.astype({"b": object, "a": object})
            changed.loc[0, ["b", "a"]] = [first, second]
            changed.to_csv(self.path, index=False)
            with self.subTest(first=first, second=second), self.assertRaises(ValueError):
                audit_submission(task, self.test, self.path)

    def test_single_probability_does_not_require_unit_row_sum(self):
        task = replace(self.task, problem_type="classification", metric="roc_auc",
                       prediction_kind="probability", classes=["a", "b"])
        self.write([.1, .3, .9])
        self.assertEqual(audit_submission(task, self.test, self.path)["status"], "ok")
        self.write([.1, 1.01, .9])
        with self.assertRaises(ValueError):
            audit_submission(task, self.test, self.path)

    def test_manifest_has_both_atomic_artifacts_and_only_safe_summary(self):
        self.write([1., 2., 3.])
        summary = {"status": "completed", "output": str(self.path), "refit_completed": True,
                   "fallback_only": False, "experiments": 2, "elapsed_seconds": np.float64(3.),
                   "OPENAI_API_KEY": "secret-marker", "environment": {"token": "secret-marker"},
                   "selected": [{"hypothesis": "private-row-marker"}], "llm_tokens": float("nan")}
        directory = self.root / "run"
        manifest = write_run_manifest(directory, self.task, summary)
        first = (directory / "manifest.json").read_text(encoding="utf-8")
        second = (self.root / "submission.manifest.json").read_text(encoding="utf-8")
        self.assertEqual(first, second)
        self.assertEqual(json.loads(first), manifest)
        self.assertEqual(manifest["status"], "ok")
        self.assertTrue(manifest["run"]["refit_completed"])
        self.assertNotIn("llm_tokens", manifest["run"])
        self.assertNotIn("secret-marker", first)
        self.assertNotIn("private-row-marker", first)
        self.assertNotIn('"0001"', first)
        self.assertEqual(manifest["submission"]["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        source = Path(__file__).resolve().parents[1] / "submission_audit.py"
        self.assertEqual(manifest["source_sha256"][source.name], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertFalse(list(self.root.rglob("*.tmp")))

    def test_manifest_reports_failed_audit_without_exposing_bad_value(self):
        self.write([1., "private-invalid-value", 3.])
        manifest = write_run_manifest(self.root / "run", self.task, {"status": "completed"})
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["submission"]["error_type"], "ValueError")
        self.assertNotIn("private-invalid-value", json.dumps(manifest))
        self.assertTrue(self.path.exists())


if __name__ == "__main__":
    unittest.main()
