"""Проверки изоляции, масок и frozen split без обучения и без Docker."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fixtures import make_task
from tools.isolated_trials import (MEMORY_BYTES, _audit_container, _mask_phase,
                                   _read_json, _validate_config, _write_masked_csv,
                                   _clean_generated_bytecode, create_trial, run_phase)


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.task = make_task(self.source, temporal=True)

    def tearDown(self):
        self.temporary.cleanup()

    def create(self, label="forest"):
        return create_trial(self.root / "cases", label, self.task,
                            {"model": "extra_trees", "seed": 42}, [ROOT / "contracts.py"])

    def test_masking_preserves_exact_raw_features_and_allowed_target_cells(self):
        original = self.root / "raw.csv"
        rows = [["id", "x", "description", "y"],
                ["0001", "0.12345678901234567890123", "comma, newline\nand quote\"", "1.000000000000001"],
                ["0002", "00123", "NA", "2.50e-12"]]
        with original.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        output = self.root / "masked.csv"
        self.assertEqual(_write_masked_csv(original, output, "y", [0], 0), 1)
        with output.open(encoding="utf-8", newline="") as stream:
            actual = list(csv.reader(stream))
        self.assertEqual(actual[0], rows[0])
        self.assertEqual(actual[1], rows[1])
        self.assertEqual(actual[2][:-1], rows[2][:-1])
        self.assertEqual(actual[2][-1], rows[1][-1])

    def test_cache_cleanup_only_removes_own_registered_bytecode(self):
        case = self.create()
        cache = case / "__pycache__"; cache.mkdir()
        (cache / "contracts.cpython-312.pyc").write_bytes(b"generated")
        _clean_generated_bytecode(case, _read_json(case / "manifest.json"))
        self.assertFalse(cache.exists())
        self.assertTrue((case / "contracts.py").exists())
        cache.mkdir(); (cache / "valuable.csv").write_text("keep", encoding="utf-8")
        with self.assertRaises(ValueError):
            _clean_generated_bytecode(case, _read_json(case / "manifest.json"))
        self.assertTrue((cache / "valuable.csv").exists())

    def test_case_never_contains_true_excluded_labels_or_foreign_files(self):
        (self.source / "GPT2.txt").write_text("чужой ответ", encoding="utf-8")
        (self.source / "foreign_model.pkl").write_bytes(b"foreign")
        case = self.create()
        self.assertEqual(pd.read_csv(case / "train.csv").response.nunique(), 1)
        self.assertFalse((case / "GPT2.txt").exists())
        self.assertFalse((case / "foreign_model.pkl").exists())
        manifest = _read_json(case / "manifest.json")
        self.assertFalse(manifest["original_train_mounted"])
        self.assertNotIn(str(self.source), json.dumps(manifest))
        contract = _read_json(case / "task_contract.json")
        self.assertEqual(contract["train_path"], "/task/train.csv")
        self.assertEqual(contract["task_path"], "/task/task.md")
        for path in (case / "splits").glob("*.npz"):
            with np.load(path, allow_pickle=False) as split:
                self.assertEqual(set(split.files), {"train", "valid"})

    def test_phase_masks_preserve_only_allowed_labels_and_never_reinfer_split(self):
        case = self.create()
        original = pd.read_csv(self.source / "train.csv")
        for phase in ("search_early", "search_late", "confirm", "full"):
            with np.load(case / "splits" / (phase + ".npz"), allow_pickle=False) as values:
                tr, va = values["train"].copy(), values["valid"].copy()
            phase_dir = case / "phases" / phase
            phase_dir.mkdir(parents=True)
            audit = _mask_phase(case, phase, self.source / "train.csv")
            masked = pd.read_csv(case / "train.csv")
            allowed = np.union1d(tr, va) if phase.startswith("search") else tr
            excluded = np.setdiff1d(np.arange(len(original)), allowed)
            np.testing.assert_allclose(masked.response.iloc[allowed], original.response.iloc[allowed])
            if len(excluded):
                np.testing.assert_allclose(masked.response.iloc[excluded], original.response.iloc[tr[0]])
            pd.testing.assert_frame_equal(masked.drop(columns="response"), original.drop(columns="response"))
            self.assertEqual(audit["allowed_label_rows"], len(allowed))
            self.assertEqual(audit["label_blinded"], phase == "confirm")
            with np.load(case / "splits" / (phase + ".npz"), allow_pickle=False) as frozen:
                np.testing.assert_array_equal(frozen["train"], tr)
                np.testing.assert_array_equal(frozen["valid"], va)

    def test_all_variants_receive_identical_splits(self):
        first, second = self.create("first"), self.create("second")
        for name in ("search_early", "search_late", "confirm", "full"):
            with np.load(first / "splits" / (name + ".npz")) as a, np.load(second / "splits" / (name + ".npz")) as b:
                np.testing.assert_array_equal(a["train"], b["train"])
                np.testing.assert_array_equal(a["valid"], b["valid"])

    def test_classification_contract_and_splits_survive_masked_labels(self):
        task = make_task(self.root / "classification", classification=True)
        case = create_trial(self.root / "cases", "classification", task,
                            {"model": "lightgbm", "seed": 42}, [ROOT / "contracts.py"])
        contract = _read_json(case / "task_contract.json")
        expected_classes = sorted(pd.read_csv(task.parent / "train.csv").category_to_predict.unique(), key=str)
        self.assertEqual(contract["classes"], expected_classes)
        (case / "phases" / "confirm").mkdir(parents=True)
        _mask_phase(case, "confirm", task.parent / "train.csv")
        with np.load(case / "splits" / "confirm.npz") as confirm:
            confirm_indices = confirm["valid"]
        for phase in ("search_main", "search_other"):
            with np.load(case / "splits" / (phase + ".npz")) as split:
                self.assertEqual(np.intersect1d(np.r_[split["train"], split["valid"]], confirm_indices).size, 0)
        self.assertEqual(_read_json(case / "task_contract.json")["classes"], expected_classes)

    def test_container_cannot_start_until_actual_mounts_are_verified(self):
        case = self.create()
        calls = []
        def fake_docker(command, **kwargs):
            calls.append(command)
            if command[1:3] == ["image", "inspect"]:
                return SimpleNamespace(stdout="sha256:fixed\n", returncode=0)
            if command[1] == "inspect":
                # Даже полностью валидные другие поля не должны разрешить лишний bind.
                info = {"Image": "sha256:fixed", "HostConfig": {}, "Mounts": [
                    {"Type": "bind", "Source": str(case), "Destination": "/task", "RW": True},
                    {"Type": "bind", "Source": str(self.source), "Destination": "/foreign", "RW": False}]}
                return SimpleNamespace(stdout=json.dumps([info]), returncode=0)
            return SimpleNamespace(stdout="created\n", returncode=0)
        with patch("tools.isolated_trials.subprocess.run", side_effect=fake_docker):
            with self.assertRaisesRegex(ValueError, "ровно один bind"):
                run_phase(case, "confirm", fixed_iterations=15, trusted_train_path=self.source / "train.csv")
        self.assertFalse(any(command[1] == "start" for command in calls))
        self.assertTrue(any(command[1:3] == ["rm", "--force"] for command in calls))

    def test_changed_original_and_missing_fixed_count_are_rejected_before_docker(self):
        case = self.create()
        with patch("tools.isolated_trials.subprocess.run") as launch:
            with self.assertRaises(ValueError):
                run_phase(case, "confirm", trusted_train_path=self.source / "train.csv")
            launch.assert_not_called()
        (case / "phases" / "search_early").mkdir(parents=True)
        with (self.source / "train.csv").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaises(ValueError):
            _mask_phase(case, "search_early", self.source / "train.csv")

    def test_configs_cannot_reference_files_scores_or_unsafe_labels(self):
        for config in ({"path": "elsewhere"}, {"base_config": {"score": .6}},
                       {"hypothesis": "E:/other/predictions.npz"}, {"hypothesis": "../runs/a"}):
            with self.assertRaises(ValueError):
                _validate_config(config)
        for label in ("../bad", "other/case", "other\\case"):
            with self.assertRaises(ValueError):
                self.create(label)
        self.create()
        with self.assertRaises(FileExistsError):
            self.create()

    def test_attestation_rejects_foreign_mount_network_and_secret_environment(self):
        case = self.create()
        info = {"Image": "sha256:fixed", "Mounts": [{"Type": "bind", "Source": str(case), "Destination": "/task", "RW": True}],
                "Config": {"Env": ["PATH=/usr/local/bin", "HOME=/tmp/home"]},
                "HostConfig": {"NetworkMode": "none", "NanoCpus": 8_000_000_000, "Memory": MEMORY_BYTES,
                               "MemorySwap": MEMORY_BYTES, "ReadonlyRootfs": True, "Privileged": False,
                               "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"],
                               "Tmpfs": {"/tmp": "rw,size=2g", "/agent": "rw,noexec,size=1m"}}}
        audit = _audit_container(info, case, "container")
        self.assertTrue(audit["attested_before_fit"])
        for change in ("mount", "network", "secret", "oldcode", "memory"):
            changed = deepcopy(info)
            if change == "mount":
                changed["Mounts"].append({"Type": "bind", "Source": str(self.root), "Destination": "/home", "RW": False})
            elif change == "network":
                changed["HostConfig"]["NetworkMode"] = "bridge"
            elif change == "secret":
                changed["Config"]["Env"].append("OPENAI_API_KEY=forbidden")
            elif change == "oldcode":
                del changed["HostConfig"]["Tmpfs"]["/agent"]
            else:
                changed["HostConfig"]["MemorySwap"] *= 2
            with self.subTest(change=change), self.assertRaises(ValueError):
                _audit_container(changed, case, "container")


if __name__ == "__main__":
    unittest.main()
