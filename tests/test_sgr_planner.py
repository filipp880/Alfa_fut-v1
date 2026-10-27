from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import json
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, MagicMock, patch

from contracts import ExperimentSpec, TaskSpec
from sgr_planner import (SGREnvelope, SGRPlanner, SGRValidationError,
                         make_sgr_schema, validate_sgr_response)


class SGRPlannerTests(unittest.TestCase):
    def test_real_request_path_passes_strict_schema_to_sdk_without_network(self):
        self.budget.until.return_value=90
        self.planner.key="test-only-not-a-real-key"
        client=MagicMock();client.__enter__.return_value=client
        client.chat.completions.create.return_value=SimpleNamespace(
            usage=SimpleNamespace(total_tokens=130,completion_tokens=80),
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.response())),finish_reason="stop")])
        with patch("openai.OpenAI",return_value=client),patch.dict("os.environ",{"OPENAI_MODEL":"qwen/qwen3.8-27b"}):
            result=self.planner.propose(self.task,{"missing_train":{"x":.2}},[])
        self.assertIsNotNone(result)
        kwargs=client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["response_format"]["type"],"json_schema")
        self.assertIs(kwargs["response_format"]["json_schema"]["strict"],True)
        self.assertEqual(kwargs["response_format"]["json_schema"]["schema"],make_sgr_schema(self.task))
        self.assertEqual(kwargs["max_tokens"],1000)
        self.assertFalse(kwargs["extra_body"]["reasoning"]["enabled"])
        self.assertTrue(kwargs["messages"][-1]["content"].endswith("/no_think"))
        self.assertEqual(self.planner.calls,1)
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "task.md").write_text("Основная метрика — RMSE.\n", encoding="utf-8")
        self.task = TaskSpec(str(self.root / "task.md"), "train.csv", "test.csv", None, None,
                             "target", "regression", "rmse", ["id"], ["x", "y", "category"],
                             ["category"], ["id", "target"])
        self.journal = Mock()
        self.budget = Mock()
        self.budget.remaining.return_value = 125.8
        self.planner = SGRPlanner(self.journal, self.budget, max_calls=1)

    def tearDown(self):
        self.temp.cleanup()

    def response(self, task=None):
        task = task or self.task
        hypothesis = "Индикатор пропуска x может улучшить качество."
        return {"observations": [{"feature": "x", "signal": "missingness",
                                  "evidence": "В переданном профиле есть пропуски x."}],
                "hypothesis": hypothesis, "risk": "low",
                "verification": {"metric": task.metric, "split": "fixed_search", "scale": "original"},
                "experiment": asdict(ExperimentSpec(missing_indicators=True, hypothesis=hypothesis))}

    def reject(self, path, value, response=None, task=None):
        candidate = deepcopy(response or self.response())
        current = candidate
        for name in path[:-1]:
            current = current[name]
        current[path[-1]] = value
        with self.assertRaises(SGRValidationError):
            validate_sgr_response(candidate, task or self.task)

    def test_schema_closes_every_object_and_requires_every_property(self):
        schema = make_sgr_schema(self.task)

        def visit(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertIs(node["additionalProperties"], False)
                    self.assertEqual(set(node["properties"]), set(node["required"]))
                for child in node.values():
                    visit(child)
            elif isinstance(node, list):
                for child in node:
                    visit(child)

        visit(schema)
        experiment = schema["properties"]["experiment"]
        self.assertEqual(set(experiment["properties"]), set(asdict(ExperimentSpec())))
        self.assertEqual(experiment["properties"]["model"]["enum"], ["catboost", "lightgbm", "linear"])

    def test_valid_response_becomes_typed_envelope(self):
        response = self.response()
        envelope = validate_sgr_response(response, self.task)
        self.assertIsInstance(envelope, SGREnvelope)
        self.assertIsInstance(envelope.experiment, ExperimentSpec)
        self.assertEqual(envelope.observations[0].feature, "x")
        self.assertEqual(envelope.verification.metric, "rmse")
        self.assertEqual(envelope.to_dict(), response)

    def test_standard_experiment_validator_is_also_called(self):
        response = self.response()
        with patch.object(ExperimentSpec, "from_dict", wraps=ExperimentSpec.from_dict) as parse:
            validate_sgr_response(response, self.task)
        parse.assert_called_once_with(response["experiment"], self.task)

    def test_unknown_and_missing_fields_are_rejected_at_each_object(self):
        for path in [(), ("verification",), ("experiment",), ("observations", 0)]:
            with self.subTest(path=path):
                candidate = self.response()
                obj = candidate
                for name in path:
                    obj = obj[name]
                obj["unexpected"] = "unused"
                with self.assertRaises(SGRValidationError):
                    validate_sgr_response(candidate, self.task)
                obj.pop("unexpected")
                obj.pop(next(iter(obj)))
                with self.assertRaises(SGRValidationError):
                    validate_sgr_response(candidate, self.task)

    def test_observation_feature_types_and_lengths_are_checked(self):
        invalid = [(("observations",), {}), (("observations", 0), None),
                   (("observations", 0, "feature"), "target"),
                   (("observations", 0, "feature"), "id"),
                   (("observations", 0, "feature"), 1),
                   (("observations", 0, "signal"), ""),
                   (("observations", 0, "signal"), "x" * 81),
                   (("observations", 0, "evidence"), []),
                   (("observations", 0, "evidence"), "x" * 161)]
        for path, value in invalid:
            with self.subTest(path=path, value=value):
                self.reject(path, value)
        observation = self.response()["observations"][0]
        self.reject(("observations",), [observation] * 4)
        candidate = self.response()
        candidate["observations"] = []
        validate_sgr_response(candidate, self.task)

    def test_summary_is_short_and_risk_and_verification_are_fixed(self):
        for path, value in [(("hypothesis",), ""), (("hypothesis",), "x" * 241),
                            (("hypothesis",), None), (("risk",), "unknown"),
                            (("risk",), {}), (("verification", "metric"), "mae"),
                            (("verification", "split"), "confirmation"),
                            (("verification", "scale"), "log")]:
            with self.subTest(path=path, value=value):
                self.reject(path, value)

    def test_experiment_requires_full_current_dsl_and_strict_types(self):
        invalid = {"model": "history_router", "iterations": 50.0, "depth": True,
                   "missing_indicators": 1, "frequency_encoding": "false", "drop_time": None,
                   "target_transform": "sqrt", "seed": False, "hypothesis": 1,
                   "learning_rate": float("nan"), "regularization": float("inf"),
                   "class_weight_power": .1, "leaves": 128, "min_child_samples": 9}
        for name, value in invalid.items():
            with self.subTest(name=name):
                self.reject(("experiment", name), value)
        self.reject(("experiment", "learning_rate"), 10 ** 1000)
        self.reject(("experiment", "hypothesis"), "x" * 1201)
        for model in ("proposal", "extra_trees", "kernel", "retrieval"):
            with self.subTest(model=model):
                self.reject(("experiment", "model"), model)

    def test_experiment_numeric_bounds_match_current_contract(self):
        bounds = {"iterations": (50, 6000), "depth": (3, 9),
                  "learning_rate": (.01, .3), "regularization": (.01, 100),
                  "leaves": (7, 127), "min_child_samples": (10, 300),
                  "seed": (0, 2 ** 31 - 1)}
        for name, (low, high) in bounds.items():
            for value in (low, high):
                with self.subTest(name=name, boundary=value):
                    candidate = self.response()
                    candidate["experiment"][name] = value
                    validate_sgr_response(candidate, self.task)
            for value in (low - 1, high + 1):
                with self.subTest(name=name, outside=value):
                    self.reject(("experiment", name), value)

    def test_transforms_use_only_known_features_and_fixed_arities(self):
        invalid = [[{"op": "ratio", "columns": ["x"]}],
                   [{"op": "missing", "columns": ["x", "y"]}],
                   [{"op": "ratio", "columns": ["x", "target"]}],
                   [{"op": "sum", "columns": ["x", 1]}],
                   [{"op": "eval", "columns": ["x"]}],
                   [{"op": "log1p", "columns": ["x"], "code": "unused"}],
                   [{"op": "missing"}], [{"op": "missing", "columns": ["x"]}] * 9]
        for transforms in invalid:
            with self.subTest(transforms=transforms):
                self.reject(("experiment", "transforms"), transforms)
        candidate = self.response()
        candidate["experiment"]["transforms"] = [
            {"op": op, "columns": ["x", "y"] if op in {"ratio", "difference", "sum"} else ["x"]}
            for op in ("ratio", "difference", "sum", "log1p", "missing")]
        validate_sgr_response(candidate, self.task)

    def test_classification_constraints_follow_experiment_spec(self):
        task = replace(self.task, problem_type="classification", metric="macro_f1", classes=["a", "b"])
        candidate = self.response(task)
        candidate["experiment"]["class_weight_power"] = 1
        validate_sgr_response(candidate, task)
        self.reject(("experiment", "target_transform"), "log1p", response=candidate, task=task)
        self.reject(("experiment", "class_weight_power"), 1.01, response=candidate, task=task)
        self.assertEqual(make_sgr_schema(task)["properties"]["verification"]["properties"]["metric"]["enum"],
                         ["macro_f1"])

    def test_empty_feature_set_allows_empty_summary_without_invalid_enum(self):
        task = replace(self.task, feature_columns=[])
        candidate = self.response()
        candidate["observations"] = []
        validate_sgr_response(candidate, task)
        candidate["observations"] = self.response()["observations"]
        with self.assertRaises(SGRValidationError):
            validate_sgr_response(candidate, task)

    def test_non_object_response_is_rejected(self):
        for value in (None, [], "{}", 0, True):
            with self.subTest(value=value), self.assertRaises(SGRValidationError):
                validate_sgr_response(value, self.task)

    def test_propose_passes_schema_and_only_its_supplied_history(self):
        history = [{"config": {"model": "linear"}, "status": "ok", "metrics": {"rmse": i},
                    "wall_seconds": 1, "feature_importance": {"x": 1}, "error_type": None,
                    "other_planner_answer": "must be excluded"} for i in range(10)]
        profile = {"missing_fraction": {"x": .2}}
        with patch.object(self.planner, "request", return_value=self.response()) as request:
            result = self.planner.propose(self.task, profile, history)
        self.assertIsInstance(result, ExperimentSpec)
        request.assert_called_once()
        system, payload = request.call_args.args
        self.assertIn("операционная сводка", system)
        self.assertEqual(request.call_args.kwargs, {"response_schema": make_sgr_schema(self.task)})
        self.assertEqual(payload["profile"], profile)
        self.assertEqual([item["metrics"]["rmse"] for item in payload["history"]], list(range(2, 10)))
        self.assertTrue(all("other_planner_answer" not in item for item in payload["history"]))
        self.assertEqual(payload["remaining_seconds"], 125)
        self.assertIsNotNone(self.planner.last_envelope)
        self.assertEqual(result.to_dict(), self.planner.last_envelope.experiment.to_dict())
        self.assertTrue(any(call.args[0] == "sgr_summary" for call in self.journal.event.call_args_list))

    def test_first_call_has_no_injected_history(self):
        with patch.object(self.planner, "request", return_value=self.response()) as request:
            self.planner.propose(self.task, {}, [])
        self.assertEqual(request.call_args.args[1]["history"], [])

    def test_invalid_response_returns_none_and_logs_only_typed_error(self):
        candidate = self.response()
        candidate["unexpected"] = "private-invalid-response-body"
        with patch.object(self.planner, "request", return_value=candidate):
            self.assertIsNone(self.planner.propose(self.task, {}, []))
        self.assertIsNone(self.planner.last_envelope)
        self.journal.event.assert_called_once_with("sgr_invalid_response", error_type="SGRValidationError")
        self.assertNotIn("private-invalid-response-body", str(self.journal.event.call_args))

    def test_none_response_clears_previous_summary(self):
        with patch.object(self.planner, "request", side_effect=[self.response(), None]):
            self.assertIsNotNone(self.planner.propose(self.task, {}, []))
            self.assertIsNone(self.planner.propose(self.task, {}, []))
        self.assertIsNone(self.planner.last_envelope)


if __name__ == "__main__":
    unittest.main()
