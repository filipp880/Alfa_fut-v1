from pathlib import Path
import tempfile
import unittest

from aide_search import BoundedSearch, default_actions, validate_actions
from contracts import ExperimentSpec
from fixtures import make_task
from task_reader import load_task


class AideTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.task, *_ = load_task(make_task(Path(self.temp.name)))
        self.config = ExperimentSpec()

    def tearDown(self):
        self.temp.cleanup()

    def search(self, policy="tree", **kwargs):
        return BoundedSearch(self.task, self.config, {"status": "ok", "score": 10.,
            "config_key": self.config.key}, policy, **kwargs)

    def test_equal_attempt_budget_same_actions_unique_configs_and_trace(self):
        traces = []
        for policy in ("linear", "tree"):
            search = self.search(policy, seed=43)
            seen = {self.config.key}
            for i in range(6):
                config = search.next()
                self.assertIsNotNone(config)
                self.assertNotIn(config.key, seen)
                seen.add(config.key)
                search.observe({"status": "ok", "score": 11.+i,
                                "wall_seconds": 1., "config_key": config.key})
            self.assertIsNone(search.next())
            trace = search.to_dict()
            self.assertEqual(trace["attempts"], 6)
            self.assertEqual(trace["wall_seconds"], 6.)
            self.assertEqual(trace["best_node_id"], 0)
            self.assertEqual(trace["extra_llm_calls"], 0)
            traces.append(trace)
        self.assertEqual(traces[0]["actions"], traces[1]["actions"])
        self.assertEqual(traces[0]["action_order"], traces[1]["action_order"])
        self.assertEqual(traces[0]["nodes"][-1]["parent"], 5)
        self.assertTrue(any(n["parent"] != n["id"]-1 for n in traces[1]["nodes"][2:]))

    def test_best_node_respects_metric_direction_and_tree_returns_to_it(self):
        for maximize in (False, True):
            self.task.metric = "r2" if maximize else "rmse"
            search = self.search(explore_probability=0.)
            search.next()
            search.observe({"status": "ok", "score": 11.})
            search.next()
            self.assertEqual(search.nodes[-1]["parent"], 1 if maximize else 0)
            search.observe({"status": "ok", "score": 9.})
            self.assertEqual(search.best_node["id"], 1 if maximize else 2)

    def test_one_bounded_resource_repair_and_no_repair_chain(self):
        actions = [{"field": "iterations", "value": 2400},
                   {"field": "iterations", "value": 800},
                   {"field": "regularization", "value": 10.}]
        search = self.search(actions=actions, explore_probability=0.)
        search.action_order = [0, 1, 2]
        search.next()
        search.observe({"status": "timeout", "reason": "timeout"})
        config = search.next()
        self.assertEqual(config.iterations, 800)
        self.assertEqual(search.nodes[-1]["stage"], "repair")
        self.assertEqual(search.nodes[-1]["parent"], 1)
        search.observe({"status": "timeout"})
        search.next()
        self.assertEqual(search.nodes[-1]["stage"], "improve")
        self.assertEqual(search.nodes[-1]["parent"], 0)

    def test_pending_mismatched_results_and_nonfinite_metrics_rejected(self):
        search = self.search()
        with self.assertRaises(RuntimeError):
            search.observe({"status": "ok", "score": 1.})
        search.next()
        with self.assertRaises(RuntimeError):
            search.next()
        with self.assertRaises(ValueError):
            search.observe({"status": "ok", "score": 1., "config_key": "wrong"})
        search.observe({"status": "ok", "score": float("nan")})
        self.assertEqual(search.nodes[-1]["status"], "invalid_metric")
        self.assertEqual(search.best_node["id"], 0)

    def test_action_schema_rejects_code_paths_targets_unknown_and_bounds(self):
        for actions in ([{"field": "code", "value": "exec"}],
                        [{"field": "target", "value": "other"}],
                        [{"field": "depth", "value": 100}],
                        [{"field": "depth", "value": True}],
                        [{"field": "learning_rate", "value": float("inf")}],
                        [{"field": "class_weight_power", "value": .5}],
                        [{"field": "iterations", "value": 100, "path": "x"}],
                        [{"field": "transforms", "value": []}], []):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                validate_actions(self.task, self.config, actions)

    def test_reproducible_seed_and_serialization_defensive_copy(self):
        one, two = self.search(seed=42), self.search(seed=42)
        for _ in range(6):
            a, b = one.next(), two.next()
            self.assertEqual(a.to_dict(), b.to_dict())
            result = {"status": "ok", "score": 12.}
            one.observe(result)
            two.observe(result)
        self.assertEqual(one.to_dict(), two.to_dict())
        copied = one.to_dict()
        copied["nodes"][0]["config"]["depth"] = 999
        self.assertEqual(one.nodes[0]["config"]["depth"], 6)
        self.assertGreaterEqual(len(default_actions(self.task, self.config)), 6)


if __name__ == "__main__":
    unittest.main()
