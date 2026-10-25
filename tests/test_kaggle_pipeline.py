"""Независимые интеграционные проверки запрета утечек через Runner."""
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

from contracts import ExperimentSpec
from data_io import encode_target, profile_data
from fixtures import make_task
from kaggle_specs import AugmentSpec
from novel_models import NovelSpec
from runtime import Budget, Runner
from task_reader import load_task
from validation import make_splits


class KagglePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def setup_case(self):
        path = make_task(self.root, temporal=True)
        frame = pd.read_csv(self.root/"train.csv", dtype={"id": str})
        frame.loc[::7, "x"] = np.nan
        frame.to_csv(self.root/"train.csv", index=False)
        task, train, test, _, dictionary = load_task(path)
        y = encode_target(task, train)
        profile, groups, times = profile_data(task, train, test, dictionary)
        confirm = make_splits(task, y, groups, times)[1]
        split_path = self.root/"confirm.npz"
        np.savez_compressed(split_path, train=confirm.train, valid=confirm.valid)
        return task, train, test, profile, confirm, split_path

    def config(self, task, base):
        return AugmentSpec(base_config=base.to_dict(),
            feature_schema={"impute_columns": ["x"], "imputer_estimators": 8,
                "pca_groups": [{"name": "numeric", "columns": ["x", "period_year"], "components": 2}]},
            history_schema={"time_column": "period_year", "entity_columns": ["category"],
                "feature_columns": task.feature_columns, "forbidden_columns": task.id_columns+[task.target]})

    def predictions(self, result):
        self.assertEqual(result["status"], "ok", result)
        with np.load(Path(result["directory"])/"predictions.npz", allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}

    def test_confirm_target_perturbation_cannot_change_pipeline_predictions(self):
        task, train, test, profile, confirm, split_path = self.setup_case()
        config = self.config(task, ExperimentSpec(iterations=50, depth=3))
        runner = Runner(self.root/"jobs", task, profile, Budget(time.monotonic(), 90, 1, 2048),
                        worker_module="kaggle_worker")
        first = runner.run(config, split_path, "confirm_first", 35, refit=True, fixed_iterations=25)
        initial = self.predictions(first)
        changed = train.copy()
        changed.loc[changed.index[confirm.valid], task.target] += 10000.
        changed.to_csv(task.train_path, index=False)
        second = runner.run(config, split_path, "confirm_changed", 35, refit=True, fixed_iterations=25)
        repeated = self.predictions(second)
        for key in ("valid", "test", "valid_indices"):
            np.testing.assert_array_equal(initial[key], repeated[key])
        self.assertNotEqual(first["score"], second["score"])
        self.assertEqual(first["best_iterations"], 25)
        self.assertEqual(second["best_iterations"], 25)
        np.testing.assert_array_equal(initial["valid_indices"], confirm.valid)
        self.assertEqual(first["n_train"], len(confirm.train))
        self.assertEqual(len(initial["test"]), len(test))
        info = first["diagnostics"]["features"]["imputation"]["x"]
        self.assertFalse(any(c.startswith("__history_") for c in info["predictors"]))
        self.assertFalse(set(task.id_columns+[task.target]) & set(info["predictors"]))

    def test_extra_trees_augmentation_exact_split_and_empty_full_validation(self):
        task, train, test, profile, confirm, _ = self.setup_case()
        config = self.config(task, NovelSpec(forest_trees=16, forest_depth=8, min_leaf=2))
        full_path = self.root/"full.npz"
        np.savez_compressed(full_path, train=np.arange(len(train)), valid=np.array([], dtype=int))
        runner = Runner(self.root/"forest_jobs", task, profile, Budget(time.monotonic(), 60, 1, 2048),
                        worker_module="kaggle_worker")
        result = runner.run(config, full_path, "full", 40, refit=True, fixed_iterations=1)
        output = self.predictions(result)
        self.assertEqual(result["n_train"], len(train))
        self.assertEqual(len(output["valid"]), 0)
        self.assertEqual(len(output["test"]), len(test))
        self.assertTrue(np.isfinite(output["test"]).all())

    def test_preflight_rejects_target_id_unknown_and_invalid_schema(self):
        task, *_ = self.setup_case()
        base = ExperimentSpec().to_dict()
        bad = [AugmentSpec(base_config=base, feature_schema={"arbitrary_code": "exec"}),
               AugmentSpec(base_config=base, feature_schema={"impute_columns": [task.target]}),
               AugmentSpec(base_config=base, feature_schema={"impute_columns": task.id_columns}),
               AugmentSpec(base_config=base, feature_schema={"impute_columns": ["unknown"]})]
        config = self.config(task, ExperimentSpec())
        config.history_schema["min_count"] = True
        bad.append(config)
        for config in bad:
            with self.subTest(config=config), self.assertRaises((ValueError, TypeError)):
                config.validate(task)


if __name__ == "__main__":
    unittest.main()
