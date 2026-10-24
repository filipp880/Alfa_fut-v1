from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from fixtures import make_task


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from package_solution import build, MODULES


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def launch(self, directory, extra=()):
        environment = os.environ.copy()
        environment.pop("OPENAI_API_KEY", None)
        environment["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run([sys.executable, str(directory/"solution.py"), "task.md",
                                 "--time-limit", "70", "--max-experiments", "2",
                                 "--max-iterations", "100", "--threads", "1", "--no-llm", *extra],
                                cwd=directory, env=environment, capture_output=True, text=True,
                                encoding="utf-8", timeout=85)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        summaries = sorted(directory.glob(".alfa_runs/*/summary.json"))
        self.assertTrue(summaries)
        return json.loads(summaries[-1].read_text(encoding="utf-8")), result.stdout

    def unpack(self, classification=False, probabilities=False):
        destination = self.root/"task"
        make_task(destination, classification=classification, probabilities=probabilities)
        archive = build(ROOT, self.root/"agent.zip")
        with zipfile.ZipFile(archive) as z:
            self.assertEqual(z.namelist(), list(MODULES))
            self.assertTrue(all('/' not in name and '\\' not in name for name in z.namelist()))
            z.extractall(destination)
        return destination

    def test_exact_flat_archive_regression(self):
        destination = self.unpack()
        summary, logs = self.launch(destination)
        self.assertEqual(summary["status"], "completed", logs)
        self.assertFalse(summary["fallback_only"])
        self.assertEqual(summary["successful_experiments"], 2, logs)
        self.assertTrue(summary["refit_completed"])
        submission = pd.read_csv(destination/"submission.csv", dtype={"id": str})
        test = pd.read_csv(destination/"test.csv", dtype={"id": str})
        self.assertEqual(submission.columns.tolist(), ['id', 'response'])
        self.assertEqual(submission.id.tolist(), test.id.tolist())
        self.assertTrue(np.isfinite(submission.response).all())

    def test_flat_archive_multiclass_probabilities(self):
        destination = self.unpack(classification=True, probabilities=True)
        summary, logs = self.launch(destination)
        self.assertEqual(summary["status"], "completed", logs)
        submission = pd.read_csv(destination/"submission.csv")
        probability = submission.drop(columns=['id']).to_numpy()
        self.assertTrue((probability >= 0).all() and (probability <= 1).all())
        np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-7)

    def test_failed_workers_keep_valid_fallback(self):
        destination = self.unpack()
        # Имитируем повреждённый ML-исполнитель, управляющий процесс остаётся настоящим.
        (destination/"worker.py").write_text("def main(path):\n raise RuntimeError('simulated training failure')\n")
        summary, _ = self.launch(destination)
        self.assertEqual(summary["status"], "fallback_no_successful_experiments")
        result = pd.read_csv(destination/"submission.csv")
        self.assertEqual(len(result), 50)
        self.assertTrue(np.isfinite(result.response).all())

    def test_build_reproducible_and_excludes_secrets(self):
        a, b = build(ROOT, self.root/"a.zip"), build(ROOT, self.root/"b.zip")
        self.assertEqual(a.read_bytes(), b.read_bytes())
        with zipfile.ZipFile(a) as z:
            self.assertNotIn(".env", z.namelist())
            self.assertNotIn("APIkey.txt", z.namelist())
            self.assertNotIn("requirements-dev.txt", z.namelist())

    def test_total_deadline_also_limits_parent_preprocessing(self):
        destination = self.unpack()
        with (destination/"data_io.py").open("a", encoding="utf-8") as stream:
            stream.write("\n_original_profile = profile_data\ndef profile_data(*args, **kwargs):\n"
                         " import time\n time.sleep(30)\n return _original_profile(*args, **kwargs)\n")
        environment = os.environ.copy()
        environment.pop("OPENAI_API_KEY", None)
        environment["PYTHONIOENCODING"] = "utf-8"
        started = time.monotonic()
        process = subprocess.run([sys.executable, str(destination/"solution.py"), "task.md",
                                  "--time-limit", "10", "--no-llm"], cwd=destination,
                                 env=environment, capture_output=True, timeout=15)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertLess(time.monotonic()-started, 13)
        frame = pd.read_csv(destination/"submission.csv")
        self.assertEqual(len(frame), 50)
        self.assertTrue(np.isfinite(frame.response).all())


if __name__ == "__main__":
    unittest.main()
