"""OOF isolation, budgets, simplex grid, and actual tiny fits; no API/public data."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from contracts import ExperimentSpec, TaskSpec
from data_io import encode_target, feature_groups, profile_data
from metrics import evaluate
from novel_models import NovelSpec
from oof_ensemble import (OOFError, _score, fit_simplex_weights, make_oof_splits,
                          propose_oof_plan, select_family_candidates)
from runtime import Budget, Runner
from validation import Split


def fixture(root, classification=False):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(26)
    n = 132
    labels = np.tile(np.arange(3), n//3)
    x = labels + rng.normal(0, .05, n) if classification else np.linspace(-2, 2, n)
    frame = pd.DataFrame({"id": [f"{i:04}" for i in range(n)], "x": x,
                          "category": np.where(x > 1, "large", "small")})
    target = np.asarray(["a", "b", "c"])[labels] if classification else 4*x+20+.1*np.sin(x)
    train, test = frame.iloc[:120].copy(), frame.iloc[120:].copy()
    train["y"] = target[:120]
    test.loc[test.index[0], "category"] = "unseen"
    task = TaskSpec(str(root/"task.md"), str(root/"train.csv"), str(root/"test.csv"), None,
                    str(root/"dictionary.csv"), "y", "classification" if classification else "regression",
                    "macro_f1" if classification else "rmse", ["id"], ["x", "category"], ["category"],
                    ["id", "y"], "label" if classification else "value", ["a", "b", "c"] if classification else [])
    dictionary = pd.DataFrame({"feature": ["id", "x", "category", "y"],
                               "role": ["id", "feature", "feature", "target"]})
    train.to_csv(task.train_path, index=False)
    test.to_csv(task.test_path, index=False)
    dictionary.to_csv(task.dictionary_path, index=False)
    Path(task.task_path).write_text("Macro F1" if classification else "RMSE", encoding="utf-8")
    y = encode_target(task, train)
    split = Split("search", np.arange(90), np.arange(90, 105), "held-out search and confirm")
    return task, train, test, dictionary, y, feature_groups(task, train), split


def candidates(task):
    specs = [ExperimentSpec(iterations=50, depth=3),
             NovelSpec(forest_trees=16, forest_depth=4, min_leaf=2),
             ExperimentSpec(model="linear", iterations=50)]
    return [{"status": "ok", "phase": "search", "config": config.to_dict(), "config_key": config.key,
             "score": .5 if task.classification else 2., "best_iterations": 7 if i == 0 else 1,
             "n_train": 90, "wall_seconds": 2., "training_seconds": .2, "prediction_seconds": .1}
            for i, config in enumerate(specs)]


class OOFFunctionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task, self.train, self.test, self.dictionary, self.y, self.groups, self.split = fixture(self.root)

    def test_selects_one_best_boosting_family_and_excludes_other_operators(self):
        rows = candidates(self.task)
        lgb = ExperimentSpec(model="lightgbm", iterations=50)
        rows += [{**rows[0], "config": lgb.to_dict(), "config_key": lgb.key, "score": 1.},
                 {**rows[0], "config": {"model": "history_router"}, "score": 0.},
                 {**rows[0], "config": {"model": "proposal"}, "score": 0.},
                 {**rows[0], "phase": "confirm", "score": -.1}]
        selected = select_family_candidates(self.task, rows)
        self.assertEqual([r["config"]["model"] for r in selected], ["lightgbm", "extra_trees", "linear"])
        rows[-4]["config"]["depth"] = 9
        self.assertNotEqual(selected[0]["config"]["depth"], 9)
        self.assertEqual(select_family_candidates(self.task, rows[:2]), [])

    def test_group_split_honors_even_a_single_duplicate_below_one_percent(self):
        y = np.arange(1000, dtype=float)
        groups = np.arange(1000)
        groups[1] = groups[0]
        split = Split("search", np.arange(1000), np.empty(0, dtype=int), "tiny duplicate fraction")
        folds = make_oof_splits(self.task, y, groups, None, split)
        self.assertEqual(len(folds), 3)
        self.assertEqual(len(np.unique(np.concatenate([f.valid for f in folds]))), 1000)
        for fold in folds:
            self.assertEqual(np.intersect1d(groups[fold.train], groups[fold.valid]).size, 0)
            self.assertIn("Group", fold.reason)

    def test_stratified_groups_rare_classes_and_unique_stratification(self):
        task = replace(self.task, problem_type="classification", metric="macro_f1", classes=["a", "b", "c"])
        y = np.tile(np.arange(3), 20)
        split = Split("search", np.arange(60), np.empty(0, dtype=int), "inner")
        for groups in (np.arange(60), np.repeat(np.arange(30), 2)):
            folds = make_oof_splits(task, y, groups, None, split)
            self.assertGreaterEqual(len(folds), 2)
            for fold in folds:
                np.testing.assert_array_equal(np.unique(y[fold.train]), np.arange(3))
                self.assertEqual(np.intersect1d(groups[fold.train], groups[fold.valid]).size, 0)
        self.assertEqual(make_oof_splits(task, y, y, None, split), [])
        rare = y.copy(); rare[rare == 2] = 0; rare[-1] = 2
        self.assertEqual(make_oof_splits(task, rare, np.arange(60), None, split), [])

    def test_temporal_expanding_folds_cover_only_later_prefix_inside_search_train(self):
        times = np.repeat(np.arange(8), 10)
        y = np.arange(80, dtype=float)
        split = Split("search", np.arange(60), np.arange(60, 70), "confirm excluded")
        folds = make_oof_splits(self.task, y, np.arange(80), times, split)
        self.assertEqual(len(folds), 3)
        covered = np.concatenate([f.valid for f in folds])
        self.assertEqual(len(covered), len(np.unique(covered)))
        self.assertTrue(set(covered) < set(split.train))
        self.assertFalse(np.intersect1d(covered, split.valid).size)
        for fold in folds:
            self.assertLess(times[fold.train].max(), times[fold.valid].min())
        self.assertEqual(make_oof_splits(self.task, y, np.tile(np.arange(10), 8), times, split), [])
        self.assertEqual(make_oof_splits(self.task, y, np.arange(80), times,
                         Split("few", np.arange(20), np.arange(20, 30), "two periods")), [])

    def test_invalid_outer_indices_are_rejected(self):
        for tr, va in ((np.array([0, 0, 1, 2]), np.array([3])), (np.arange(10), np.array([0])),
                       (np.array([0., 1., 2., 3.]), np.array([4])), (np.array([-1, 0, 1, 2]), np.array([3]))):
            self.assertEqual(make_oof_splits(self.task, self.y, self.groups, None, Split("bad", tr, va, "")), [])

    def test_grid_contains_equal_thirds_quarter_pairs_and_allows_zero_weights(self):
        matrix = np.array([[1., 0., -1.], [0., 1., -1.], [1., 0., -1.], [0., 1., -1.]])
        equal = fit_simplex_weights(self.task, np.zeros(4), matrix)
        np.testing.assert_allclose(equal["weights"], [1/3]*3)
        self.assertAlmostEqual(equal["score"], 0.)
        self.assertEqual(equal["evaluated"], 73)
        pair = fit_simplex_weights(self.task, np.zeros(4), np.tile([1., -3., 10.], (4, 1)))
        np.testing.assert_array_equal(pair["weights"], [.75, .25, 0.])
        vertex = fit_simplex_weights(self.task, np.zeros(4), np.tile([1., 2., 0.], (4, 1)))
        np.testing.assert_array_equal(vertex["weights"], [0., 0., 1.])

    def test_classification_grid_and_fast_metrics_match_standard_original_axis(self):
        task = replace(self.task, problem_type="classification", metric="macro_f1", classes=["a", "b", "c"])
        y = np.tile(np.arange(3), 5)
        rng = np.random.default_rng(4)
        p = rng.random((len(y), 3));p /= p.sum(axis=1, keepdims=True)
        for metric in ("macro_f1", "weighted_f1", "accuracy", "balanced_accuracy"):
            spec = replace(task, metric=metric)
            self.assertAlmostEqual(_score(spec, y, p), evaluate(spec, y, p)[metric])
        matrix = np.stack([p, np.eye(3)[y], np.full_like(p, 1/3)], axis=1)
        fitted = fit_simplex_weights(task, y, matrix)
        np.testing.assert_array_equal(fitted["weights"], [0., 1., 0.])
        binary = replace(task, classes=["positive", "negative"], metric="f1", positive_index=0)
        self.assertAlmostEqual(_score(binary, y % 2, p[:, :2]/p[:, :2].sum(axis=1, keepdims=True)),
                               evaluate(binary, y % 2, p[:, :2]/p[:, :2].sum(axis=1, keepdims=True))["f1"])

    def test_invalid_matrix_shapes_nan_and_probability_axes_fail_closed(self):
        for matrix in (np.array(1.), np.zeros((4, 2)), np.full((4, 3), np.nan)):
            with self.assertRaises(OOFError):
                fit_simplex_weights(self.task, np.zeros(4), matrix)
        task = replace(self.task, problem_type="classification", metric="macro_f1", classes=["a", "b", "c"])
        for matrix in (np.ones((4, 3, 2))/2, np.ones((4, 3, 3)), np.full((4, 3, 3), np.nan)):
            with self.assertRaises(OOFError):
                fit_simplex_weights(task, np.zeros(4, dtype=int), matrix)

    def test_expired_or_partial_grid_never_returns_partial_weights(self):
        with patch("oof_ensemble.time.monotonic", return_value=100.):
            with self.assertRaisesRegex(OOFError, "weight_grid_deadline"):
                fit_simplex_weights(self.task, np.zeros(4), np.zeros((4, 3)), deadline=99.)
        with patch("oof_ensemble.time.monotonic", side_effect=[0., 0., 100.]):
            with self.assertRaisesRegex(OOFError, "weight_grid_deadline"):
                fit_simplex_weights(self.task, np.zeros(4), np.zeros((4, 3)), deadline=1.)


class OOFControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task, self.train, self.test, self.dictionary, self.y, self.groups, self.split = fixture(self.root/"data")
        self.budget = SimpleNamespace(seconds=10000, until=lambda fraction: 500., threads=1, memory_mb=2048)
        self.runner = SimpleNamespace(profile={"time_column": None}, worker_module="worker")
        self.calls = []
        self.bad = None
        self.advance = None

    def factory(self, directory, task, profile, budget, **kwargs):
        owner = self
        class FakeRunner:
            def run(self, config, split_path, phase, timeout, refit=False, fixed_iterations=None):
                with np.load(split_path, allow_pickle=False) as values:
                    tr, va = values["train"].copy(), values["valid"].copy()
                owner.calls.append({"tr": tr, "va": va, "profile": profile, "budget": budget,
                                    "refit": refit, "fixed": fixed_iterations, "timeout": timeout})
                work = Path(directory)/f"model_{len(owner.calls)}";work.mkdir()
                bias = {"catboost": .3, "lightgbm": .3, "extra_trees": -.2, "linear": .05}[config.model]
                valid = 4*owner.train.x.iloc[va].to_numpy()+20+bias+.001*profile["target_summary"]["mean"]
                predicted = np.full(len(owner.test), 20.)
                actual = va.copy()
                result = {"status": "ok", "directory": str(work), "config_key": config.key,
                          "best_iterations": fixed_iterations, "n_train": len(tr),
                          "score": evaluate(task, owner.y[va], valid)[task.metric]}
                if owner.bad == "indices": actual = actual[::-1]
                if owner.bad == "nan": valid[0] = np.nan
                if owner.bad == "shape": valid = valid[:, None]
                if owner.bad == "test_shape": predicted = predicted[:-1]
                if owner.bad == "count": result["best_iterations"] += 1
                if owner.bad == "score": result["score"] += 100
                if owner.bad == "fit_eval": result["evaluation_used_for_fit"] = True
                if owner.bad == "stopped": result["training_stopped_for_budget"] = True
                if owner.bad == "failed": result["status"] = "failed"
                np.savez_compressed(work/"predictions.npz", valid=valid, test=predicted, valid_indices=actual)
                if owner.advance is not None: owner.advance()
                return result
        return FakeRunner()

    def propose(self, name="run"):
        with patch("oof_ensemble.Runner", side_effect=self.factory):
            return propose_oof_plan(self.task, self.train, self.test, self.dictionary, self.y, self.groups,
                None, self.split, candidates(self.task), self.runner, self.budget, self.root/name, Mock())

    def test_all_members_share_oof_indices_fixed_counts_and_fold_only_profiles(self):
        plan = self.propose()
        self.assertIsNotNone(plan)
        self.assertEqual(len(self.calls), 9)
        for call in self.calls:
            self.assertTrue(call["refit"])
            self.assertIn(call["fixed"], (1, 7))
            self.assertIs(call["budget"], self.budget)
            self.assertEqual(call["profile"]["train_rows"], len(call["tr"]))
            self.assertEqual(call["profile"]["profile_scope"], "fold_train_only")
            self.assertAlmostEqual(call["profile"]["target_summary"]["mean"], self.y[call["tr"]].mean())
            self.assertFalse(np.intersect1d(np.r_[call["tr"], call["va"]], self.split.valid).size)
            self.assertTrue(np.isin(np.r_[call["tr"], call["va"]], self.split.train).all())
        status = json.loads((self.root/"run/oof/status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "completed")
        self.assertEqual([status[k] for k in ("fit_attempts", "completed_fits", "validated_fits")], [9, 9, 9])
        with np.load(self.root/"run/oof/raw_matrix.npz", allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw["indices"], self.split.train)
            self.assertEqual(raw["predictions"].shape, (90, 3))
            mixed = sum(w*raw["predictions"][:, i] for i, w in enumerate(status["weights"]))
            self.assertAlmostEqual(plan["oof_score"], evaluate(self.task, self.y[self.split.train], mixed)[self.task.metric])

    def test_search_valid_confirm_and_test_target_perturbations_cannot_change_weights(self):
        first = self.propose("first")
        excluded = np.setdiff1d(np.arange(len(self.y)), self.split.train)
        self.y[excluded] += 1e9
        self.train.loc[excluded, "y"] = self.y[excluded]
        self.test["y"] = -1e30
        second = self.propose("second")
        self.assertEqual(first, second)

    def test_any_invalid_fold_discards_entire_plan_without_a_partial_matrix(self):
        for mode in ("indices", "nan", "shape", "test_shape", "count", "score", "fit_eval", "stopped", "failed"):
            with self.subTest(mode=mode):
                self.bad = mode
                self.assertIsNone(self.propose(mode))
                out = self.root/mode/"oof"
                self.assertEqual(json.loads((out/"status.json").read_text())["status"], "discarded")
                self.assertFalse((out/"raw_matrix.npz").exists())

    def test_budget_estimate_skips_before_fits_and_slow_fold_discards_all(self):
        self.budget.until = lambda fraction: 20.
        self.assertIsNone(self.propose("low"))
        self.assertEqual(len(self.calls), 0)
        self.budget.until = lambda fraction: 500.
        clock = [100.]
        self.advance = lambda: clock.__setitem__(0, clock[0]+600.)
        with patch("oof_ensemble.time.monotonic", side_effect=lambda: clock[0]):
            self.assertIsNone(self.propose("slow"))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(json.loads((self.root/"slow/oof/status.json").read_text())["status"], "discarded")

    def test_invalid_fold_proof_is_rechecked_before_any_worker(self):
        folds = [Split("bad0", np.arange(60), np.arange(60, 75), ""),
                 Split("bad1", np.arange(60), np.arange(60, 75), "")]
        with patch("oof_ensemble.make_oof_splits", return_value=folds):
            self.assertIsNone(self.propose("proof"))
        self.assertEqual(self.calls, [])

    def test_matrix_memory_guard_skips_before_any_worker(self):
        self.budget.memory_mb = .01
        self.assertIsNone(self.propose("memory"))
        self.assertEqual(self.calls, [])
        status = json.loads((self.root/"memory/oof/status.json").read_text())
        self.assertEqual(status["reason"], "matrix_memory_budget")
        self.assertEqual(status["fit_attempts"], 0)
        self.assertGreater(status["estimated_peak_bytes"], status["memory_cap_bytes"])

    def test_parent_time_column_is_frozen_after_fold_profiling(self):
        self.runner.profile["time_column"] = "x"
        self.assertIsNotNone(self.propose("frozen_time_column"))
        self.assertTrue(all(call["profile"]["time_column"] == "x" for call in self.calls))


class OOFActualSmallFitTests(unittest.TestCase):
    def test_actual_fixed_fits_ignore_changed_validation_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            task,train,test,dictionary,y,groups,split=fixture(root/'data')
            profile,_,_=profile_data(task,train.iloc[split.train],test,dictionary)
            budget=Budget(time.monotonic(),600,1,2048)
            split_path=root/'split.npz'
            np.savez_compressed(split_path,train=split.train,valid=split.valid)
            configs=[ExperimentSpec(model='linear'),ExperimentSpec(model='lightgbm',iterations=50),
                     NovelSpec(forest_trees=16,forest_depth=4,min_leaf=2)]
            for number,config in enumerate(configs):
                predictions=[]
                for changed in (False,True):
                    frame=train.copy()
                    if changed:frame.loc[split.valid,'y']+=1e6
                    frame.to_csv(task.train_path,index=False)
                    runner=Runner(root/f'fit_{number}_{changed}',task,profile,budget)
                    result=runner.run(config,split_path,'oof_fixed',35,refit=True,
                                      fixed_iterations=16 if config.model=='lightgbm' else 1)
                    self.assertEqual(result['status'],'ok',result)
                    self.assertFalse(result.get('evaluation_used_for_fit',False))
                    with np.load(Path(result['directory'])/'predictions.npz') as raw:
                        predictions.append((raw['valid'].copy(),raw['test'].copy()))
                for first,second in zip(*predictions):np.testing.assert_array_equal(first,second)

    def run_case(self, classification):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task, train, test, dictionary, y, groups, split = fixture(root/"data", classification)
            profile, _, _ = profile_data(task, train.iloc[split.train], test, dictionary)
            budget = Budget(time.monotonic(), 1800, 1, 2048)
            runner = Runner(root/"search", task, profile, budget)
            split_path = root/"search.npz"
            np.savez_compressed(split_path, train=split.train, valid=split.valid)
            successes = []
            for row in candidates(task):
                config = ExperimentSpec.from_dict(row["config"], task) if row["config"]["model"] != "extra_trees" else NovelSpec.from_dict(row["config"], task)
                result = runner.run(config, split_path, "search", 30)
                self.assertEqual(result["status"], "ok", result)
                successes.append(result)
            plan = propose_oof_plan(task, train, test, dictionary, y, groups, None, split,
                                    successes, runner, budget, root/"ensemble", Mock())
            out = root/"ensemble/oof"
            status = json.loads((out/"status.json").read_text(encoding="utf-8"))
            self.assertIsNotNone(plan, status)
            self.assertEqual(status["status"], "completed")
            self.assertEqual(len(status["jobs"]), 9)
            with np.load(out/"raw_matrix.npz", allow_pickle=False) as raw:
                mixed = sum(w*raw["predictions"][:, i] for i, w in enumerate(status["weights"]))
                self.assertAlmostEqual(evaluate(task, y[raw["indices"]], mixed)[task.metric], plan["oof_score"])
                if classification:
                    self.assertEqual(raw["predictions"].shape, (90, 3, 3))
                    np.testing.assert_allclose(mixed.sum(axis=1), 1.)
            for request_path in out.glob("fold*/*/request.json"):
                request = json.loads(request_path.read_text(encoding="utf-8"))
                self.assertTrue(request["refit"])
                expected = next(r["best_iterations"] for r in successes if r["config_key"] ==
                                next(j["config_key"] for j in status["jobs"] if Path(j["directory"]).name == request_path.parent.name))
                self.assertEqual(request["fixed_iterations"], expected)
            if not classification:
                with np.load(out/"raw_matrix.npz", allow_pickle=False) as raw:
                    initial_matrix = raw["predictions"].copy()
                outside = np.setdiff1d(np.arange(len(y)), split.train)
                changed_y = y.copy();changed_y[outside] += 1e7
                changed_train = train.copy();changed_train.loc[outside, "y"] = changed_y[outside]
                changed_test = test.copy();changed_test["y"] = -1e30
                changed_train.to_csv(task.train_path, index=False)
                changed_test.to_csv(task.test_path, index=False)
                second = propose_oof_plan(task, changed_train, changed_test, dictionary, changed_y, groups,
                    None, split, successes, runner, budget, root/"perturbed_outside", Mock())
                self.assertEqual(plan, second)
                with np.load(root/"perturbed_outside/oof/raw_matrix.npz", allow_pickle=False) as raw:
                    np.testing.assert_array_equal(raw["predictions"], initial_matrix)

    def test_actual_small_regression_models_produce_complete_oof_plan(self):
        self.run_case(False)

    def test_actual_small_multiclass_models_preserve_probability_axis(self):
        self.run_case(True)


if __name__ == "__main__":
    unittest.main()
