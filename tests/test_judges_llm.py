"""LLM boundary regressions: fake SDK / local transport only, never a live API."""
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import httpx
import openai

from llm_client import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMPlanner


class JudgesLLMTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.journal = Mock()
        self.budget = Mock()
        self.budget.until.return_value = 100.
        self.budget.remaining.return_value = 150.

    def planner(self, calls=6):
        result = LLMPlanner(self.journal, self.budget, calls)
        result.key = "test-only-not-a-real-key"
        return result

    def client(self, content='{"model":"linear"}', usage=100, finish="stop"):
        client = MagicMock()
        client.__enter__.return_value = client
        client.chat.completions.create.return_value = SimpleNamespace(
            usage=SimpleNamespace(total_tokens=usage, completion_tokens=20),
            choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)])
        return client

    def events(self, name):
        return [call.kwargs for call in self.journal.event.call_args_list if call.args == (name,)]

    def test_invalid_and_blank_token_limits_do_not_break_disabled_constructor(self):
        for raw in ("", " ", "invalid", "3.5", "9" * 5000):
            with self.subTest(raw=raw[:20]), patch.dict(os.environ, {"ALFA_LLM_TOKEN_LIMIT": raw}):
                planner = LLMPlanner(self.journal, self.budget, 0)
                self.assertEqual(planner.token_limit, 40000)
                self.assertFalse(planner.available)

    def test_valid_token_limits_keep_existing_bounds(self):
        for raw, expected in (("999", 1000), (" 1200 ", 1200), ("100001", 100000)):
            with self.subTest(raw=raw), patch.dict(os.environ, {"ALFA_LLM_TOKEN_LIMIT": raw}):
                self.assertEqual(LLMPlanner(self.journal, self.budget).token_limit, expected)

    def test_malformed_metric_values_are_rejected_without_exception(self):
        planner = self.planner()
        for value in ([], {}, None, 0, True, "unknown", "", ["rmse"]):
            with self.subTest(value=value), patch.object(planner, "request", return_value={"metric": value}):
                self.assertIsNone(planner.resolve_metric(SimpleNamespace(target="y"), "task"))
        with patch.object(planner, "request", return_value={"metric": "rmse"}):
            self.assertEqual(planner.resolve_metric(SimpleNamespace(target="y"), "task"), "rmse")

    def test_default_constructor_and_qwen_request_flags_are_unchanged(self):
        planner, client = self.planner(), self.client()
        with patch("openai.OpenAI", return_value=client) as constructor:
            self.assertEqual(planner.request("system", {"history": []}), {"model": "linear"})
        self.assertEqual(constructor.call_args.kwargs, {
            "api_key": planner.key, "base_url": DEFAULT_BASE_URL, "timeout": 35, "max_retries": 0})
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], DEFAULT_MODEL)
        self.assertEqual(request["temperature"], .15)
        self.assertEqual(request["max_tokens"], 1000)
        self.assertEqual(request["extra_body"], {"enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False}, "reasoning": {"enabled": False}})
        self.assertTrue(request["messages"][1]["content"].endswith("\n/no_think"))
        self.assertEqual(planner.calls, 1)
        self.assertEqual(planner.tokens, 100)

    def test_empty_model_and_endpoint_use_known_defaults(self):
        for raw in ("", " \t "):
            with self.subTest(raw=raw), patch.dict(os.environ, {"OPENAI_MODEL": raw, "OPENAI_BASE_URL": raw}):
                client = self.client()
                with patch("openai.OpenAI", return_value=client) as constructor:
                    self.assertIsNotNone(self.planner().request("system", {}))
                self.assertEqual(constructor.call_args.kwargs["base_url"], DEFAULT_BASE_URL)
                self.assertEqual(client.chat.completions.create.call_args.kwargs["model"], DEFAULT_MODEL)

    def test_explicit_non_qwen_model_is_respected_without_qwen_options(self):
        with patch.dict(os.environ, {"OPENAI_MODEL": "vendor/model", "OPENAI_BASE_URL": "https://example.invalid/v2"}):
            client = self.client()
            with patch("openai.OpenAI", return_value=client) as constructor:
                self.planner().request("system", {})
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "vendor/model")
        self.assertNotIn("extra_body", kwargs)
        self.assertFalse(kwargs["messages"][1]["content"].endswith("/no_think"))
        self.assertEqual(constructor.call_args.kwargs["base_url"], "https://example.invalid/v2")

    def test_missing_or_whitespace_key_skips_without_sdk_construction(self):
        for raw in ("", " \n\t"):
            with self.subTest(raw=raw), patch.dict(os.environ, {"OPENAI_API_KEY": raw}):
                planner = LLMPlanner(self.journal, self.budget)
                with patch("openai.OpenAI") as constructor:
                    self.assertIsNone(planner.request("system", {}))
                    self.assertIsNone(planner.request("system", {}))
                constructor.assert_not_called()
                self.assertEqual(planner.calls, 0)
                self.assertEqual(planner.tokens, 0)
        self.assertTrue(all(not event["key_present"] for event in self.events("llm_config")))
        self.assertTrue(all(event["reason"] == "missing_key" for event in self.events("llm_skipped")))

    def test_skip_reasons_cover_disabled_time_context_token_and_call_limits(self):
        cases = (("disabled", lambda p: setattr(p, "max_calls", 0)),
                 ("search_time_limit", lambda p: setattr(self.budget.until, "return_value", 7)),
                 ("token_reserve", lambda p: setattr(p, "token_limit", 1000)),
                 ("call_limit", lambda p: setattr(p, "calls", p.max_calls)))
        for reason, configure in cases:
            with self.subTest(reason=reason):
                self.budget.until.return_value = 100.
                planner = self.planner()
                configure(planner)
                with patch("openai.OpenAI") as constructor:
                    self.assertIsNone(planner.request("system", {}))
                constructor.assert_not_called()
                self.assertEqual(self.events("llm_skipped")[-1]["reason"], reason)
        planner = self.planner()
        with patch("openai.OpenAI") as constructor:
            self.assertIsNone(planner.request("system", {"data": "x" * 24001}))
        constructor.assert_not_called()
        self.assertEqual(self.events("llm_skipped")[-1]["reason"], "context_limit")

    def test_timeout_keeps_existing_search_fraction_and_no_retries(self):
        self.budget.until.return_value = 9.
        with patch("openai.OpenAI", return_value=self.client()) as constructor:
            self.planner().request("system", {})
        self.assertEqual(constructor.call_args.kwargs["timeout"], 7.)
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)

    def test_malformed_json_keeps_usage_and_finish_reason(self):
        planner = self.planner()
        with patch("openai.OpenAI", return_value=self.client('{"model":', finish="length")):
            self.assertIsNone(planner.request("system", {}))
        self.assertEqual(planner.tokens, 100)
        error = self.events("llm_error")[-1]
        self.assertEqual(error["stage"], "parse")
        self.assertEqual(error["finish_reason"], "length")
        self.assertEqual(self.events("llm_usage")[-1]["usage_source"], "reported")

    def test_missing_reported_usage_uses_estimate_and_preserves_valid_response(self):
        planner = self.planner()
        with patch("openai.OpenAI", return_value=self.client(usage=None)):
            self.assertEqual(planner.request("system", {}), {"model": "linear"})
        self.assertGreater(planner.tokens, 0)
        self.assertEqual(self.events("llm_usage")[-1]["usage_source"], "estimated")

    def test_error_and_config_logs_do_not_copy_credentials_query_or_error_body(self):
        secret = "private-secret-marker"
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://user:password@example.invalid/v1?token=" + secret}):
            planner = self.planner()
            client = self.client()
            response = httpx.Response(400, request=httpx.Request("POST", "https://example.invalid"))
            client.chat.completions.create.side_effect = openai.BadRequestError(
                "Authorization: Bearer " + secret, response=response, body={"token": secret})
            with patch("openai.OpenAI", return_value=client):
                self.assertIsNone(planner.request("system", {}))
        events = repr(self.journal.event.call_args_list)
        for value in (secret, "password", "user:", planner.key, "Authorization"):
            self.assertNotIn(value, events)
        config = self.events("llm_config")[-1]
        self.assertEqual(config["gateway_host"], "example.invalid")
        self.assertEqual(config["gateway_path"], "/v1")
        error = self.events("llm_error")[-1]
        self.assertEqual((error["stage"], error["http_status"]), ("request", 400))

    def test_constructor_error_is_safe_and_three_errors_still_disable_requests(self):
        planner = self.planner()
        with patch("openai.OpenAI", side_effect=ValueError("private-constructor-value")) as constructor:
            for _ in range(4):
                self.assertIsNone(planner.request("system", {}))
        self.assertEqual(constructor.call_count, 3)
        self.assertEqual(planner.calls, 3)
        self.assertEqual(self.events("llm_error")[-1]["stage"], "client_init")
        self.assertEqual(self.events("llm_skipped")[-1]["reason"], "error_limit")
        self.assertNotIn("private-constructor-value", repr(self.journal.event.call_args_list))

    def test_key_inside_model_or_gateway_path_is_redacted_only_in_diagnostics(self):
        secret = "DummyPrivateCredential42"
        endpoint = "https://" + secret + ".example.invalid/v1/" + secret
        with patch.dict(os.environ, {"OPENAI_MODEL": secret, "OPENAI_BASE_URL": endpoint}):
            planner = self.planner()
            planner.key = secret
            client = self.client()
            with patch("openai.OpenAI", return_value=client) as constructor:
                self.assertIsNotNone(planner.request("system", {}))
        self.assertEqual(constructor.call_args.kwargs["api_key"], secret)
        self.assertEqual(constructor.call_args.kwargs["base_url"], endpoint)
        self.assertEqual(client.chat.completions.create.call_args.kwargs["model"], secret)
        self.assertNotIn(secret.casefold(), repr(self.journal.event.call_args_list).casefold())
        config = self.events("llm_config")[-1]
        self.assertEqual(config["model"], "<redacted>")
        self.assertEqual(config["gateway_host"], "<redacted>.example.invalid")
        self.assertEqual(config["gateway_path"], "/v1/<redacted>")

    def test_actual_sdk_timeout_is_not_retried_and_logs_only_stage(self):
        requests = []
        def timeout(request):
            requests.append(request)
            raise httpx.ReadTimeout("private-timeout-body", request=request)
        real_constructor = openai.OpenAI
        def local_constructor(**kwargs):
            return real_constructor(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(timeout)))
        planner = self.planner()
        with patch("openai.OpenAI", side_effect=local_constructor):
            self.assertIsNone(planner.request("system", {}))
        self.assertEqual(len(requests), 1)
        self.assertEqual(planner.calls, 1)
        self.assertGreater(planner.tokens, 0)
        error = self.events("llm_error")[-1]
        self.assertEqual((error["error_type"], error["stage"]), ("APITimeoutError", "request"))
        self.assertNotIn("private-timeout-body", repr(self.journal.event.call_args_list))

    def test_actual_sdk_serialization_uses_local_transport_only(self):
        requests = []
        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"id": "offline", "object": "chat.completion", "created": 0,
                "model": DEFAULT_MODEL, "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"model":"linear"}'}}],
                "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}})
        real_constructor = openai.OpenAI
        def local_constructor(**kwargs):
            return real_constructor(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        planner = self.planner()
        with patch("openai.OpenAI", side_effect=local_constructor):
            self.assertEqual(planner.request("system", {"history": []}), {"model": "linear"})
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(str(request.url), DEFAULT_BASE_URL + "/chat/completions")
        body = json.loads(request.content)
        self.assertEqual(body["model"], DEFAULT_MODEL)
        self.assertFalse(body["enable_thinking"])
        self.assertEqual(body["reasoning"], {"enabled": False})
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("extra_body", body)
        self.assertTrue(body["messages"][1]["content"].endswith("/no_think"))


if __name__ == "__main__":
    unittest.main()
