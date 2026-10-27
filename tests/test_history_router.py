"""Проверки точного G6, фиксации seed и восстановления production-рецепта."""
from dataclasses import replace
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from contracts import ExperimentSpec, TaskSpec
from historical_features import HISTORY_COLUMNS, HistoricalFeatures, HistorySchema
from history_router import (HistoryRouterSpec, RouterNumericPolicy,
                            history_schema_diagnostics, predict_router,
                            suggest_router_schema, train_router)
from models import train_model
from novel_models import AlternativeModel, NovelSpec
from preprocessing import FeatureBuilder


def router_case():
    records = []
    for period in range(1, 5):
        for j, entity in enumerate("abcd"):
            records.append({"id": str(len(records)), "period": period, "entity": entity,
                            "x": float(j + period), "z": np.nan if j == 0 else float(j),
                            "label": float(10*j + period**2)})
    frame = pd.DataFrame(records)
    frame.loc[13, "entity"] = "new"
    frame.loc[15, "entity"] = None
    task = TaskSpec(task_path="unused/task.md", train_path="unused/train.csv",
                    test_path="unused/test.csv", sample_path=None, dictionary_path=None,
                    target="label", problem_type="regression", metric="rmse",
                    id_columns=["id"], feature_columns=["period", "entity", "x", "z"],
                    categorical_columns=["entity"], output_columns=["id", "label"])
    test = pd.DataFrame({"id": ["x0", "x1", "x2", "x3", "x4"],
                         "period": [5, 5, 5, 1, 2], "entity": ["a", "new", None, "a", "b"],
                         "x": [2., 3., 4., 5., 0.], "z": [np.nan, 0., 1., 2., np.nan]})
    schema = {"time_column": "period", "entity_columns": ["entity"]}
    return task, frame, test, frame.label.to_numpy(), np.arange(12), np.arange(12, 16), schema


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


class FastForest:
    def __init__(self, task, config, threads):
        self.config = config

    def fit(self, frame, y):
        self.n_train_ = len(frame)
        return self

    def predict(self, frame):
        return np.full(len(frame), self.config.seed + 100., dtype=float)


class HistoryRouterTests(unittest.TestCase):
    def test_huber_reserve_matches_independent_frozen_composition_and_roundtrip(self):
        from proposal_models import ProposalSpec, train_proposal
        task, train, test, y, tr, va, schema = router_case()
        native = HistoryRouterSpec(history_schema=schema, depth=3, iterations=50)
        combined = replace(native, fallback="forest_huber", fallback_iterations=50, fallback_weight=.5)
        profile = {"time_column": "period"}
        original, base_valid, base_test, _ = train_router(task, native, profile, train, test, y, tr, va, 1, True, 8, 60)
        bundle, valid, predicted, details = train_router(task, combined, profile, train, test, y, tr, va, 1, True, 8, 60)
        config = ProposalSpec(variant="huber", base_config=ExperimentSpec(model="lightgbm", iterations=50).to_dict())
        _, hv, ht, _ = train_proposal(task, config, profile, train, test, y, tr, va, 1, True, 50, 60)
        masks = [original["history"].transform(frame)["__history_known"].to_numpy(dtype=bool)
                 for frame in (train.iloc[va], test)]
        expected_valid = np.where(masks[0], base_valid, .5*base_valid+.5*hv)
        expected_test = np.where(masks[1], base_test, .5*base_test+.5*ht)
        np.testing.assert_array_equal(valid, expected_valid)
        np.testing.assert_array_equal(predicted, expected_test)
        np.testing.assert_array_equal(predict_router(pickle.loads(pickle.dumps(bundle)), test), predicted)
        self.assertEqual(details["component_fits"], 3)
        self.assertEqual(details["component_iterations"]["huber"], [50])
    def test_spec_is_bounded_and_rejects_unsupported_or_forbidden_routes(self):
        task, _, _, _, _, _, schema = router_case()
        config = HistoryRouterSpec(history_schema=schema)
        self.assertEqual((config.depth, config.iterations, config.learning_rate), (8, 2400, .045))
        self.assertEqual(HistoryRouterSpec.from_dict(config.to_dict(), task).key, config.key)
        self.assertEqual(replace(config, hypothesis="Другое объяснение").key, config.key)
        self.assertNotEqual(replace(config, seed_count=2).key, config.key)
        invalid = [replace(config, history_policy="exec"), replace(config, fallback="huber_global"),
                   replace(config, fallback="huber_unknown"), replace(config, seed_count=True),
                   replace(config, seed_count=3), replace(config, seed_count=2, seed=2**31-1),
                   replace(config, history_schema={"time_column": "period", "entity_columns": ["id"]}),
                   replace(config, history_schema={"time_column": "period", "entity_columns": ["label"]}),
                   replace(config, history_schema={"time_column": "period", "entity_columns": ["unknown"]}),
                   replace(config, history_schema={**schema, "code": "exec"})]
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises((ValueError, TypeError)):
                candidate.validate(task)
        with self.assertRaises(ValueError):
            config.validate(replace(task, problem_type="classification"))

    def test_native_router_exactly_reproduces_previous_all_train_g6(self):
        task, train, test, y, tr, va, schema = router_case()
        config = HistoryRouterSpec(history_schema=schema, depth=3, iterations=50)
        profile = {"time_column": "period"}
        bundle, valid, predicted, details = train_router(
            task, config, profile, train, test, y, tr, va, 1,
            refit=True, fixed_iterations=8, time_limit=60)

        old_history = HistoricalFeatures(HistorySchema(
            **schema, feature_columns=task.feature_columns, forbidden_columns=["id", "label"]))
        xtr = old_history.fit_transform(train.iloc[tr][task.feature_columns], y[tr])
        xva = old_history.transform(train.iloc[va][task.feature_columns])
        xte = old_history.transform(test[task.feature_columns])
        augmented = train.copy()
        for column in HISTORY_COLUMNS:
            augmented[column] = np.nan
            augmented.loc[tr, column] = xtr[column].to_numpy()
            augmented.loc[va, column] = xva[column].to_numpy()
        augmented_test = test.copy()
        for column in HISTORY_COLUMNS:
            augmented_test[column] = xte[column].to_numpy()
        augmented_task = replace(task, feature_columns=list(xtr.columns))
        base = ExperimentSpec(depth=3, iterations=50, learning_rate=.045)
        _, old_valid, old_test, _ = train_model(
            augmented_task, base, profile, augmented, augmented_test, y, tr, va, 1,
            refit=True, fixed_iterations=8, time_limit=60)
        forest = AlternativeModel(task, NovelSpec(forest_trees=256, forest_depth=24,
                                                  min_leaf=4, max_features=1.), 1).fit(train.iloc[tr], y[tr])
        np.testing.assert_array_equal(valid, np.where(xva["__history_known"].astype(bool), old_valid, forest.predict(train.iloc[va])))
        np.testing.assert_array_equal(predicted, np.where(xte["__history_known"].astype(bool), old_test, forest.predict(test)))
        self.assertEqual(details["n_train"], len(tr))
        self.assertEqual(details["diagnostics"]["members"][0]["n_train"], len(tr))
        self.assertEqual(bundle["history_models"][0]["model"].tree_count_, 8)
        self.assertEqual(details["best_iterations"], 8)
        self.assertEqual(details["component_fits"], 2)
        self.assertEqual(details["component_iterations"], {"history": [8], "forest": [256]})
        self.assertFalse(details["evaluation_used_for_fit"])
        self.assertEqual(details["diagnostics"]["history"]["known_history_rows"], 8)

        restored = pickle.loads(pickle.dumps(bundle, protocol=4))
        np.testing.assert_array_equal(predict_router(restored, test), predicted)
        duplicate_index = test.iloc[[4, 1, 0, 0]].copy()
        duplicate_index.index = [7, 7, 2, 2]
        np.testing.assert_array_equal(predict_router(restored, duplicate_index), predicted[[4, 1, 0, 0]])
        self.assertEqual(predict_router(restored, test.iloc[:0]).shape, (0,))

    def test_confirm_labels_do_not_change_predictions_and_full_refit_works(self):
        task, train, test, y, tr, va, schema = router_case()
        config = HistoryRouterSpec(history_schema=schema, depth=3, iterations=50, history_policy="median")
        first, a, b, _ = train_router(task, config, {"time_column": "period"}, train, test, y, tr, va, 1,
                                     True, 5, 60)
        self.assertEqual(first["policy"].columns, ["z"])
        self.assertNotIn("x__was_missing", first["history_models"][0]["builder"].columns)
        self.assertNotIn("period__was_missing", first["history_models"][0]["builder"].columns)
        changed, changed_y = train.copy(), y.copy()
        changed_y[va] += 1e7
        changed.loc[va, "label"] = changed_y[va]
        _, c, d, _ = train_router(task, config, {"time_column": "period"}, changed, test, changed_y, tr, va, 1,
                                  True, 5, 60)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
        np.testing.assert_array_equal(first["history"].transform(train.iloc[tr])["__history_known"],
                                      [0]*4 + [1]*8)
        full, empty, full_predicted, details = train_router(
            task, config, {"time_column": "period"}, train, test, y, np.arange(len(train)),
            np.array([], dtype=int), 1, True, 5, 60)
        self.assertEqual(len(empty), 0)
        self.assertEqual(details["n_train"], len(train))
        self.assertTrue(np.isfinite(full_predicted).all())
        np.testing.assert_array_equal(predict_router(pickle.loads(pickle.dumps(full)), test), full_predicted)

    def test_second_seed_uses_primary_iteration_count_without_early_stopping(self):
        task, train, test, y, tr, va, schema = router_case()
        calls = []

        def fake_train(task, config, profile, frame, test, target, tr, va, threads,
                       refit=False, fixed_iterations=None, time_limit=None):
            calls.append({"seed": config.seed, "refit": refit, "fixed": fixed_iterations,
                          "train_rows": len(tr), "history_known": frame.iloc[tr]["__history_known"].to_numpy()})
            builder = FeatureBuilder(task, config, profile).fit(frame.iloc[tr])
            bundle = {"builder": builder, "model": ConstantModel(config.seed),
                      "task": task.to_dict(), "config": config.to_dict()}
            return bundle, np.full(len(va), config.seed), np.full(len(test), config.seed), {
                "best_iterations": fixed_iterations or 7, "training_stopped_for_budget": False}

        config = HistoryRouterSpec(history_schema=schema, seed_count=2, iterations=50, depth=3)
        with patch("history_router.train_model", side_effect=fake_train), patch("history_router.AlternativeModel", FastForest):
            bundle, valid, predicted, details = train_router(task, config, {"time_column": "period"},
                train, test, y, tr, va, 1, time_limit=30)
        self.assertEqual([(c["seed"], c["refit"], c["fixed"]) for c in calls], [(42, False, None), (43, True, 7)])
        self.assertTrue(all(c["train_rows"] == len(tr) for c in calls))
        self.assertTrue(all(np.sum(c["history_known"]) == 8 for c in calls))
        self.assertEqual(details["best_iterations"], 7)
        self.assertEqual(details["component_fits"], 4)
        self.assertEqual(details["component_iterations"], {"history": [7, 7], "forest": [256, 256]})
        np.testing.assert_array_equal(valid, [42.5, 142.5, 42.5, 142.5])
        np.testing.assert_array_equal(predict_router(pickle.loads(pickle.dumps(bundle)), test), predicted)
        calls.clear()
        with patch("history_router.train_model", side_effect=fake_train), patch("history_router.AlternativeModel", FastForest):
            train_router(task, config, {"time_column": "period"}, train, test, y, tr, va, 1, True, 9, 30)
        self.assertEqual([(c["refit"], c["fixed"]) for c in calls], [(True, 9), (True, 9)])

    def test_policies_fit_only_original_x_keep_zero_and_do_not_fill_history(self):
        frame = pd.DataFrame({"x": [1., np.nan, 0., np.inf], "y": [0., 1., np.nan, 4.]})
        median = RouterNumericPolicy("median", ["x", "y"]).fit(frame)
        before = dict(median.medians_)
        transformed = median.transform(pd.DataFrame({"x": [np.nan, 1e30, 0.], "y": [0., np.nan, 1.]}))
        self.assertEqual(transformed.x.iloc[0], .5)
        self.assertEqual(transformed.x.iloc[2], 0.)
        self.assertEqual(median.medians_, before)
        no_missing = RouterNumericPolicy("median", ["x"]).fit(pd.DataFrame({"x": [0., 1., 2.]}))
        self.assertEqual(no_missing.columns, [])
        self.assertEqual(list(no_missing.transform(pd.DataFrame({"x": [np.nan]})).columns), ["x"])
        pattern = RouterNumericPolicy("missing_pattern", ["x", "y"]).fit(frame).transform(frame)
        np.testing.assert_array_equal(pattern["__row_missing_count"], [0., 1., 1., 1.])
        np.testing.assert_array_equal(pattern["__row_zero_count"], [1., 0., 1., 0.])
        self.assertTrue(np.isnan(pattern.x.iloc[1]))
        with self.assertRaises(ValueError):
            RouterNumericPolicy("median", ["x"]).fit(frame.assign(x__was_missing=0))

        task, train, test, y, tr, va, schema = router_case()
        config = HistoryRouterSpec(history_schema=schema, history_policy="missing_pattern", iterations=50, depth=3)
        bundle, _, predicted, _ = train_router(task, config, {"time_column": "period"},
            train, test, y, tr, va, 1, True, 4, 60)
        self.assertFalse(any(c.startswith("__history_") for c in bundle["policy"].columns))
        self.assertTrue(bundle["history"].transform(train.iloc[:4])["__history_mean"].isna().all())
        np.testing.assert_array_equal(predict_router(pickle.loads(pickle.dumps(bundle)), test), predicted)

    def test_missing_numeric_key_stays_unknown_after_median_imputation(self):
        task, train, test, y, tr, va, _ = router_case()
        config = HistoryRouterSpec(history_schema={"time_column": "period", "entity_columns": ["z"]},
                                   history_policy="median", iterations=50, depth=3)
        bundle, _, predicted, _ = train_router(task, config, {"time_column": "period"},
            train, test, y, tr, va, 1, True, 4, 60)
        known = bundle["history"].transform(test)["__history_known"].to_numpy(dtype=bool)
        self.assertFalse(known[0])
        self.assertFalse(known[4])
        forest = bundle["forest_models"][0].predict(test)
        np.testing.assert_array_equal(predicted[~known], forest[~known])

    def test_suggestion_uses_explicit_metadata_training_coverage_and_no_target(self):
        task, train, _, _, tr, _, _ = router_case()
        dictionary = pd.DataFrame({"feature": ["id", "period", "entity", "x", "z", "label"],
            "role": ["id", "feature", "feature", "feature", "feature", "target"],
            "dtype": ["category", "numeric", "category", "numeric", "numeric", "numeric"],
            "description_ru": ["Entity key", "Код периода", "Идентификатор объекта", "Измерение", "Другое", "Entity key"]})
        original = suggest_router_schema(dictionary, task, {"time_column": "period"}, train.iloc[tr])
        self.assertEqual(original["entity_columns"], ["entity"])
        changed = train.iloc[tr].copy()
        changed["label"] = ["не читается"]*len(changed)
        self.assertEqual(suggest_router_schema(dictionary, task, {"time_column": "period"}, changed), original)
        diagnostics = history_schema_diagnostics(original, changed)
        self.assertTrue(diagnostics["eligible"])
        self.assertAlmostEqual(diagnostics["coverage"], 2/3)
        self.assertNotIn("different_target_key_period_groups", diagnostics)
        new_keys = train.iloc[tr].copy()
        new_keys["entity"] = np.arange(len(new_keys)).astype(str)
        self.assertEqual(suggest_router_schema(dictionary, task, {"time_column": "period"}, new_keys), {})
        no_semantics = dictionary.copy()
        no_semantics.loc[no_semantics.feature == "entity", "description_ru"] = "Обычная категория"
        self.assertEqual(suggest_router_schema(no_semantics, task, {"time_column": "period"}, train.iloc[tr]), {})
        static = no_semantics.copy()
        static.loc[static.feature == "entity", "description_ru"] = "Статический атрибут сущности"
        self.assertEqual(suggest_router_schema(static, task, {"time_column": "period"}, train.iloc[tr])["entity_columns"], ["entity"])
        row_id = dictionary.copy()
        row_id.loc[row_id.feature == "entity", "description_ru"] = "Идентификатор строки, entity key"
        self.assertEqual(suggest_router_schema(row_id, task, {"time_column": "period"}, train.iloc[tr]), {})

    def test_existing_building_metadata_remains_supported_without_column_names(self):
        columns = ["p", "r", "c", "u", "a", "b"]
        task, *_ = router_case()
        task = replace(task, feature_columns=columns, categorical_columns=["r", "c", "u"])
        dictionary = pd.DataFrame({"feature": columns, "role": ["feature"]*6,
            "dtype": ["numeric", "category", "category", "category", "numeric", "numeric"],
            "description_ru": ["Период", "Код штата", "Класс здания", "Назначение здания", "Площадь здания", "Год постройки"]})
        frame = pd.DataFrame({"p": [1, 2, 3], "r": ["r"]*3, "c": ["c"]*3,
                              "u": ["u"]*3, "a": [30.]*3, "b": [1900.]*3})
        suggested = suggest_router_schema(dictionary, task, {"time_column": "p"}, frame)
        self.assertEqual(suggested["entity_columns"], columns[1:])
        duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        diagnostics = history_schema_diagnostics(suggested, duplicated)
        self.assertEqual(diagnostics["multirow_key_period_groups"], 1)
        self.assertEqual(diagnostics["multirow_key_period_rows"], 2)

    def test_invalid_splits_and_unfrozen_refit_fail_before_training(self):
        task, train, test, y, tr, va, schema = router_case()
        config = HistoryRouterSpec(history_schema=schema)
        for changes in ({"tr": np.array([0, 0])}, {"va": np.array([0])},
                        {"tr": np.array([-1])}, {"tr": np.array([1.5])},
                        {"refit": True}, {"fixed_iterations": True}, {"time_limit": 0}):
            kwargs = dict(task=task, config=config, profile={"time_column": "period"},
                          train=train, test=test, y=y, tr=tr, va=va, threads=1)
            kwargs.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                train_router(**kwargs)


if __name__ == "__main__":
    unittest.main()
