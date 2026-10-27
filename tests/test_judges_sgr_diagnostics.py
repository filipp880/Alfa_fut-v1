"""Reporter compatibility for sanitized LLM diagnostics; no API, fit or Docker."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from contracts import ExperimentSpec, TaskSpec
from tools.report_sgr_study import AuditError, prompt_audit


class SGRDiagnosticCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        path = self.directory / "task.md"
        path.write_text("Метрика RMSE", encoding="utf-8")
        self.task = TaskSpec(str(path), "train.csv", "test.csv", None, None,
            "target", "regression", "rmse", ["id"], ["x"], [], ["id", "target"])
        self.prompt = {"system": "исследователь", "payload": {
            "task_excerpt": "Метрика RMSE", "task": {"type": "regression", "metric": "rmse",
            "features": ["x"], "classes": []}, "profile": {"experiment_seed": 42}, "history": [],
            "remaining_seconds": 10799, "example": ExperimentSpec().to_dict()}, "response_schema": None,
            "other_policy_context": False, "previous_reports_in_context": False}
        self.ref = {"task": self.task, "profile": {}, "frozen_paths": [],
                    "system_prompts": {"normal": "исследователь"}}
        self.arm = {"policy": "normal", "seed": 42, "valid_proposal": False, "calls": 1, "tokens": 100}
        self.config = {"event": "llm_config", "model": "qwen/qwen3.8-27b",
            "gateway_host": "api.alfahack-gateway26.ru", "gateway_path": "/v1", "gateway_valid": True,
            "key_present": True, "max_calls": 1, "token_limit": 40000, "timeout_seconds": 35,
            "max_retries": 0, "max_completion_tokens": 1000, "thinking_disabled": True,
            "invalid_token_limit_defaulted": False}
        self.usage = {"event": "llm_usage", "call": 1, "tokens": 100, "total_tokens": 100,
                      "completion_tokens": 20, "finish_reason": "length", "usage_source": "reported"}
        self.error = {"event": "llm_error", "call": 1, "error_type": "JSONDecodeError",
                      "http_status": None, "stage": "parse", "finish_reason": "length"}

    def audit(self, events):
        (self.directory / "prompt.json").write_text(json.dumps(self.prompt, ensure_ascii=False), encoding="utf-8")
        (self.directory / "experiments.jsonl").write_text(
            "\n".join(json.dumps(event, ensure_ascii=False) for event in events), encoding="utf-8")
        return prompt_audit(self.directory, self.arm, self.ref)

    def test_new_diagnostics_are_validated_without_becoming_quality_results(self):
        skipped = {"event": "llm_skipped", "reason": "call_limit", "calls": 1, "tokens": 100, "errors": 1}
        result = self.audit([self.config, self.usage, self.error, skipped,
                             {"event": "llm_invalid_metric", "reason": "missing_or_invalid_metric"}])
        self.assertEqual(result["usage"]["reported_confirmed"], 100)
        self.assertEqual(result["usage"]["estimated_usage_fallback"], 0)
        self.assertFalse(any("score" in field or "quality" in field for field in result))

    def test_legacy_usage_and_error_remain_valid(self):
        legacy_usage = {k: v for k, v in self.usage.items() if k not in {"usage_source", "finish_reason"}}
        legacy_error = {k: v for k, v in self.error.items() if k not in {"stage", "finish_reason", "http_status"}}
        result = self.audit([legacy_usage, legacy_error])
        self.assertEqual(result["usage"]["reported_confirmed"], 100)

    def test_explicit_estimate_is_separate_from_reported_usage(self):
        estimate = (len(self.prompt["system"]) + len(json.dumps(self.prompt["payload"], ensure_ascii=False, default=str))) // 2 + 1400
        self.arm["tokens"] = estimate
        usage = {**self.usage, "usage_source": "estimated", "tokens": estimate, "total_tokens": estimate}
        result = self.audit([self.config, usage, self.error])
        self.assertEqual(result["usage"]["estimated_usage_fallback"], estimate)
        self.assertEqual(result["usage"]["reported_confirmed"], 0)
        usage["tokens"] = usage["total_tokens"] = estimate - 1
        self.arm["tokens"] -= 1
        with self.assertRaises(AuditError):
            self.audit([usage, self.error])

    def test_no_key_skip_requires_no_usage_or_proposal(self):
        self.arm.update(calls=0, tokens=0)
        config = {**self.config, "key_present": False}
        skipped = {"event": "llm_skipped", "reason": "missing_key", "calls": 0, "tokens": 0, "errors": 0}
        result = self.audit([config, skipped])
        self.assertEqual(result["tokens"], 0)

    def test_secret_or_quality_fields_and_unrecognized_values_are_rejected(self):
        invalid = [
            {**self.config, "api_key": "must-not-be-accepted"},
            {**self.config, "gateway_host": "user:password@api.alfahack-gateway26.ru"},
            {**self.config, "max_retries": 1},
            {**self.error, "message": "raw-error-body"},
            {**self.error, "stage": "other"},
            {**self.error, "finish_reason": ["length"]},
            {**self.error, "http_status": True},
            {**self.usage, "usage_source": "other"},
            {"event": "llm_skipped", "reason": "call_limit", "calls": 1, "tokens": 100, "errors": 1, "score": .9},
            {"event": "llm_skipped", "reason": [], "calls": 1, "tokens": 100, "errors": 1},
        ]
        for event in invalid:
            with self.subTest(event=event), self.assertRaises(AuditError):
                self.audit([event, self.usage, self.error])

    def test_diagnostics_do_not_relax_prompt_profile_history_or_system_checks(self):
        original = deepcopy(self.prompt)
        for field in ("history", "profile", "system"):
            self.prompt = deepcopy(original)
            if field == "system":
                self.prompt["system"] = "foreign prompt"
            else:
                self.prompt["payload"][field] = [{"score": .9}] if field == "history" else {"foreign_context": True}
            with self.subTest(field=field), self.assertRaises(AuditError):
                self.audit([self.config, self.usage, self.error])


if __name__ == "__main__":
    unittest.main()
