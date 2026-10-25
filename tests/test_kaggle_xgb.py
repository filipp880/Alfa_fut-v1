"""Проверки категорий, запрещённых столбцов и слепого fixed-iteration refit."""
from pathlib import Path
import importlib.util
import tempfile
import unittest

import numpy as np

from contracts import ExperimentSpec
from data_io import encode_target
from fixtures import make_task
from kaggle_xgb import XGBSpec, _make_builder, class_metrics, train_xgboost
from task_reader import load_task


class XGBoostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_config_closed_bounded_and_no_target_or_id(self):
        task, *_ = load_task(make_task(self.root))
        XGBSpec().validate(task)
        self.assertEqual(XGBSpec(hypothesis="A").key, XGBSpec(hypothesis="B").key)
        self.assertEqual(XGBSpec.from_dict(XGBSpec().to_dict(), task), XGBSpec())
        for invalid in (XGBSpec(model="os"), XGBSpec(iterations=True), XGBSpec(depth=0),
                        XGBSpec(learning_rate=float("nan")), XGBSpec(class_weight_power=.5),
                        XGBSpec(frequency_encoding=1)):
            with self.assertRaises(ValueError):
                invalid.validate(task)
        with self.assertRaises(ValueError):
            ExperimentSpec(model="xgboost").validate(task)
        task.feature_columns.append(task.target)
        with self.assertRaises(ValueError):
            XGBSpec().validate(task)

    def test_category_codes_train_only_unknown_is_missing(self):
        task, train, test, *_ = load_task(make_task(self.root))
        config = XGBSpec(frequency_encoding=True, missing_indicators=True)
        builder = _make_builder(task, config, {}, train.iloc[:120])
        levels = {k: list(v) for k, v in builder.levels.items()}
        before = builder.transform(train.iloc[:120], categorical_codes=True)
        changed = test.copy()
        changed[task.target] = 1e100
        changed[task.id_columns[0]] = "forbidden"
        after = builder.transform(changed, categorical_codes=True)
        self.assertTrue(after["category"].iloc[:2].isna().all())
        self.assertEqual(levels, builder.levels)
        self.assertEqual(list(before["category"].cat.categories), list(after["category"].cat.categories))
        self.assertFalse(set(task.id_columns+[task.target]) & set(after.columns))

    def test_per_class_report_includes_absent_and_rare_classes(self):
        task, *_ = load_task(make_task(self.root, classification=True))
        report = class_metrics(task, np.array([0, 0, 1]), np.eye(3)[[0, 1, 1]])
        self.assertEqual(report[str(task.classes[0])]["support"], 2)
        self.assertEqual(report[str(task.classes[2])]["support"], 0)
        self.assertEqual(report[str(task.classes[2])]["f1"], 0.)

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "XGBoost отсутствует в локальном dev-стенде")
    def test_regression_multiclass_and_refit_does_not_read_validation_labels(self):
        for classification in (False, True):
            task, train, test, *_ = load_task(make_task(self.root/str(classification), classification=classification))
            y = encode_target(task, train)
            tr, va = np.arange(160), np.arange(160, len(train))
            config = XGBSpec(iterations=12, depth=3, min_child_weight=1,
                             class_weight_power=.5 if classification else 0)
            bundle, valid, predicted, details = train_xgboost(
                task, config, {}, train, test, y, tr, va, 1, refit=True, fixed_iterations=9)
            changed = y.copy()
            changed[va] = 999999
            _, second, second_test, second_details = train_xgboost(
                task, config, {}, train, test, changed, tr, va, 1, refit=True, fixed_iterations=9)
            np.testing.assert_array_equal(valid, second)
            np.testing.assert_array_equal(predicted, second_test)
            self.assertFalse(details["evaluation_used_for_fit"])
            self.assertFalse(second_details["evaluation_used_for_fit"])
            self.assertEqual(details["best_iterations"], 9)
            self.assertEqual(bundle["model"].num_boosted_rounds(), 9)
            self.assertTrue(np.isfinite(predicted).all())
            if classification:
                np.testing.assert_allclose(predicted.sum(axis=1), 1.)
            _, empty, _, details = train_xgboost(task, config, {}, train, test, y, tr,
                                                np.array([], dtype=int), 1, refit=True,
                                                fixed_iterations=2)
            self.assertEqual(len(empty), 0)
            self.assertEqual(details["best_iterations"], 2)

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "XGBoost отсутствует в локальном dev-стенде")
    def test_search_uses_target_metric_and_absent_training_class_supported(self):
        task, train, test, *_ = load_task(make_task(self.root, classification=True))
        y = encode_target(task, train)
        tr = np.flatnonzero(y != 2)
        va = np.flatnonzero(y == 2)
        bundle, valid, predicted, details = train_xgboost(
            task, XGBSpec(iterations=8, depth=3), {}, train, test, y, tr, va, 1)
        self.assertTrue(details["evaluation_used_for_fit"])
        self.assertEqual(valid.shape, (len(va), 3))
        self.assertEqual(predicted.shape, (len(test), 3))
        self.assertEqual(details["search_best_score"], 0.)
        self.assertEqual(details["early_stopping_metric"], "macro_f1")
        self.assertEqual(bundle["model"].num_boosted_rounds(), details["best_iterations"])


if __name__ == "__main__":
    unittest.main()
