"""Контракт положительного класса: offline tiny CLI, без API и обучения."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from task_reader import explicit_positive_index, load_task


ROOT = Path(__file__).resolve().parents[1]


class JudgesTaskContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="alfa_task_contract_")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_case(self, labels, token, name="case"):
        root = self.root / name
        root.mkdir()
        pd.DataFrame({"id": [f"t{i}" for i in range(6)], "x": range(6),
                      "target": [labels[0]]*5 + [labels[1]]}).to_csv(root / "train.csv", index=False)
        pd.DataFrame({"id": ["000a", "NA", "000002"], "x": [7, 8, 9]}).to_csv(root / "test.csv", index=False)
        pd.DataFrame({"id": ["000a", "NA", "000002"], "target": [.5]*3}).to_csv(root / "sample_submission.csv", index=False)
        pd.DataFrame({"feature": ["id", "x", "target"], "role": ["id", "feature", "target"],
                      "dtype": ["string", "numeric", "category" if isinstance(labels[0], str) else "numeric"]}).to_csv(
                          root / "data_dictionary.csv", index=False)
        path = root / "task.md"
        path.write_text(f"Primary metric: ROC AUC.\nPositive class: {token}.\nPredict probabilities.", encoding="utf-8")
        return path

    def test_decimal_positive_class_keeps_internal_dot(self):
        for index, token in enumerate(("`0.0`", "0.0", '"0.0"', "'0.0'")):
            with self.subTest(token=token):
                task, *_ = load_task(self.make_case([0., 1.], token, str(index)))
                self.assertEqual(task.classes, [0., 1.])
                self.assertEqual(task.positive_index, 0)

    def test_dotted_and_quoted_string_labels_are_complete_tokens(self):
        cases = [("benign.case", "malicious.case", "`benign.case`"),
                 ("benign.case", "malicious.case", '"benign.case"'),
                 ("benign.case", "malicious.case", "'benign.case'"),
                 ("O'Reilly.v1", "Z.class", "`O'Reilly.v1`"),
                 ('O"Reilly.v1', "Z.class", json.dumps('O"Reilly.v1'))]
        for index, (positive, negative, token) in enumerate(cases):
            with self.subTest(positive=positive, token=token):
                task, *_ = load_task(self.make_case([positive, negative], token, str(index)))
                self.assertEqual(task.classes[task.positive_index], positive)

    def test_literal_na_and_leading_zero_labels_and_ids_are_preserved(self):
        for index, labels in enumerate((["NA", "Z"], ["000", "001"])):
            with self.subTest(labels=labels):
                task, train, test, *_ = load_task(self.make_case(labels, f"`{labels[0]}`", str(index)))
                self.assertEqual(task.classes[task.positive_index], labels[0])
                self.assertEqual(train.target.iloc[0], labels[0])
                self.assertEqual(test.id.tolist(), ["000a", "NA", "000002"])

    def test_existing_integer_zero_one_contract(self):
        task, *_ = load_task(self.make_case([0, 1], "`0`"))
        self.assertEqual(task.classes, [0, 1])
        self.assertEqual(task.positive_index, 0)
        self.assertEqual(explicit_positive_index("Positive class is 1.", task.classes), 1)
        self.assertEqual(explicit_positive_index("Положительный класс: `0`.", task.classes), 0)

    def test_exact_case_has_priority_and_ambiguous_case_is_not_guessed(self):
        task, *_ = load_task(self.make_case(["na", "NA"], "`na`"))
        self.assertEqual(task.classes[task.positive_index], "na")
        self.assertIsNone(explicit_positive_index("Positive class: `Na`.", task.classes))

    def test_prefixes_and_unknown_numeric_spelling_are_not_class_labels(self):
        self.assertIsNone(explicit_positive_index("Positive class: `benign`.", ["benign.case", "malicious.case"]))
        self.assertIsNone(explicit_positive_index("Positive class: `0`.", [0., 1.]))
        self.assertEqual(explicit_positive_index("Positive class: good.v1.", ["good.v1", "bad.v1"]), 0)
        self.assertIsNone(explicit_positive_index("Positive class: good.v2.", ["good.v1", "bad.v1"]))

    def test_real_tiny_cli_json_and_csv_use_explicit_positive_axis(self):
        path = self.make_case([0., 1.], "`0.0`")
        environment = os.environ.copy()
        environment.pop("OPENAI_API_KEY", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run([sys.executable, "-B", str(ROOT / "solution.py"), str(path),
                                 "--time-limit", "10", "--no-llm"], cwd=self.root,
                                env=environment, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        summary_path = next(path.parent.glob(".alfa_runs/*/summary.json"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "fallback_small_or_constant_task")
        self.assertEqual(summary["task"]["classes"], [0., 1.])
        self.assertEqual(summary["task"]["positive_index"], 0)
        payload = (path.parent / "submission.csv").read_bytes()
        frame = pd.read_csv(path.parent / "submission.csv", dtype={"id": str}, keep_default_na=False)
        self.assertEqual(frame.id.tolist(), ["000a", "NA", "000002"])
        np.testing.assert_allclose(frame.target, 5/6, rtol=0., atol=1e-15)
        manifest = json.loads((path.parent / "submission.manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["submission"]["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(manifest["status"], "ok")


if __name__ == "__main__":
    unittest.main()
