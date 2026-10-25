from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import time
import unittest
import zipfile

import numpy as np
from contracts import ExperimentSpec, parse_experiment
from data_io import encode_target, profile_data
from fixtures import make_task
from novel_models import AlternativeModel, MixedEncoder, NovelSpec
from runtime import Budget, Runner
from research import Research, choose_finalists
from task_reader import load_task
from validation import make_splits


class NovelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_config_is_bounded_and_standard_agent_remains_closed(self):
        task, *_ = load_task(make_task(self.root))
        NovelSpec().validate(task)
        for invalid in (NovelSpec(model="os"), NovelSpec(neighbors=True),
                        NovelSpec(components=100000), NovelSpec(kernel_scale=float("nan")),
                        NovelSpec(target_transform="log1p")):
            with self.assertRaises(ValueError):
                invalid.validate(task)
        with self.assertRaises(ValueError):
            ExperimentSpec(model="extra_trees").validate(task)
        self.assertIsInstance(parse_experiment(NovelSpec().to_dict(), task), NovelSpec)

    def test_encoder_uses_only_training_features(self):
        task, train, test, *_ = load_task(make_task(self.root))
        encoder = MixedEncoder()
        encoder.fit_transform(task, train.iloc[:120])
        before = encoder.encoder.named_transformers_["numeric"][0].statistics_.copy()
        changed = test.copy()
        changed[task.target] = 1e100
        changed[task.id_columns[0]] = "forbidden"
        changed.loc[changed.index[0], "x"] = np.inf
        changed.loc[changed.index[1], "x"] = np.nan
        encoded = encoder.transform(changed)
        self.assertTrue(np.isfinite(encoded).all())
        self.assertNotIn(task.target, encoder.features)
        self.assertFalse(set(task.id_columns) & set(encoder.features))
        np.testing.assert_array_equal(before, encoder.encoder.named_transformers_["numeric"][0].statistics_)

    def test_all_families_regression_classification_unknowns_and_empty(self):
        for classification in (False, True):
            task, train, test, *_ = load_task(make_task(self.root/str(classification), classification=classification))
            y = encode_target(task, train)
            for family in ("extra_trees", "kernel", "retrieval"):
                with self.subTest(classification=classification, family=family):
                    config = NovelSpec(model=family, forest_trees=16, components=32, neighbors=5,
                                       class_weight_power=.5 if classification else 0)
                    model = AlternativeModel(task, config, 1).fit(train.iloc[:180], y[:180])
                    predicted = model.predict(test)
                    self.assertEqual(len(predicted), len(test))
                    self.assertTrue(np.isfinite(predicted).all())
                    self.assertEqual(len(model.predict(test.iloc[:0])), 0)
                    if classification:
                        np.testing.assert_allclose(predicted.sum(axis=1), 1.)
                        self.assertTrue(((predicted >= 0) & (predicted <= 1)).all())
                    if family == "retrieval":
                        self.assertEqual(len(model.memory_y), 180)
                        np.testing.assert_array_equal(model.memory_y, y[:180])

    def test_isolated_worker_obeys_exact_split(self):
        task, train, test, _, dd = load_task(make_task(self.root, classification=True))
        y = encode_target(task, train)
        profile, groups, times = profile_data(task, train, test, dd)
        search = make_splits(task, y, groups, times)[0]
        np.savez_compressed(self.root/"split.npz", train=search.train, valid=search.valid)
        # The production worker must dispatch alternatives too.
        runner = Runner(self.root/"jobs", task, profile, Budget(time.monotonic(), 60, 1, 2048))
        result = runner.run(NovelSpec(model="retrieval", neighbors=5, class_weight_power=.5),
                            self.root/"split.npz", "search", 45)
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["n_train"], len(search.train))
        with np.load(Path(result["directory"])/"predictions.npz") as values:
            np.testing.assert_array_equal(values["valid_indices"], search.valid)
            self.assertEqual(len(values["test"]), len(test))

    def test_alternative_portfolio_is_opt_in_for_reference_controls(self):
        task, train, test, _, dd = load_task(make_task(self.root))
        profile, _, _ = profile_data(task, train, test, dd)
        planner = type("Offline", (), {"available": False})()
        original = Research(task, profile, planner)
        extended = Research(task, profile, planner, include_alternatives=True)
        self.assertFalse(any(c.model == "extra_trees" for c in original.queue))
        self.assertEqual(sum(c.model == "extra_trees" for c in extended.queue), 1)

    def test_ensemble_can_choose_complement_outside_top_two_individuals(self):
        task, *_ = load_task(make_task(self.root))
        results = [{"name": "leader", "score": 1., "config_key": "a", "config": {"model": "catboost"}},
                   {"name": "second", "score": 1.1, "config_key": "b", "config": {"model": "lightgbm"}},
                   {"name": "complement", "score": 2., "config_key": "c", "config": {"model": "extra_trees"}}]
        predictions = {"leader": np.array([1., 1., -1., -1.]),
                       "second": np.array([1.1, 1.1, -1.1, -1.1]),
                       "complement": np.array([-2., -2., 2., 2.])}
        candidates, weights = choose_finalists(task, results, np.zeros(4), lambda r: predictions[r["name"]])
        self.assertEqual([r["name"] for r in candidates], ["leader", "complement"])
        self.assertEqual(weights["c"], .75)

    def test_two_useful_complement_families_survive_search_ranking(self):
        task, *_ = load_task(make_task(self.root))
        results = [{"config_key": "a", "score": 1., "config": {"model": "catboost"}},
                   {"config_key": "b", "score": 1.4, "config": {"model": "lightgbm"}},
                   {"config_key": "c", "score": 2., "config": {"model": "extra_trees"}}]
        predictions = {"a": np.array([1., 1., -1., -1.]),
                       "b": np.array([-1.4, -1.4, 1.4, 1.4]),
                       "c": np.array([-2., -2., 2., 2.])}
        candidates, weights = choose_finalists(task, results, np.zeros(4), lambda r: predictions[r["config_key"]])
        self.assertEqual([r["config_key"] for r in candidates], ["a", "b", "c"])
        self.assertEqual(weights, {"b": .5, "c": .75})

    def test_research_zip_is_flat_reproducible_and_requirements_unchanged(self):
        from tools.run_unconventional import package
        first, second = package(self.root/"first.zip"), package(self.root/"second.zip")
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as archive:
            self.assertFalse(any("/" in name or "\\" in name for name in archive.namelist()))
            self.assertTrue({"novel_models.py", "novel_worker.py", "unconventional.py", "solution.py"}
                            <= set(archive.namelist()))
            self.assertEqual(hashlib.sha256(archive.read("requirements.txt")).hexdigest(),
                             "9173b0388532fbd2a709e4f4b3029e3b9ccab685fce43a85261ffa1ba95efbe6")


if __name__ == "__main__":
    unittest.main()
