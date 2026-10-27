"""Журнал и атомарные JSON-артефакты. В журнале нет API-ключей."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from contracts import native


def atomic_json(path: Path, data: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(native(data), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Journal:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "experiments.jsonl"

    def event(self, event: str, **data):
        record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **native(data)}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        public_fields = (
            "event", "phase", "model", "score", "status", "reason", "error_type",
            "http_status", "stage", "finish_reason", "call", "calls", "tokens", "total_tokens", "errors",
            "usage_source", "completion_tokens", "content_characters", "elapsed_seconds",
            "gateway_host", "gateway_path", "gateway_valid", "key_present", "max_calls",
            "token_limit", "timeout_seconds", "max_retries", "max_completion_tokens",
            "thinking_disabled", "invalid_token_limit_defaulted", "prediction_source",
            "refit_completed", "fallback_only", "llm_calls", "llm_tokens",
            "submission_rows", "submission_sha256",
        )
        message = {k: record[k] for k in public_fields if k in record}
        if event == "experiment_result" and record.get("status") != "ok" and isinstance(record.get("error"), str):
            # Только ML-исключение из worker без API-секретов. Тело ошибки
            # сетевого LLM-клиента никогда не печатается этой веткой.
            message["error_message"] = record["error"].replace("\r", " ").replace("\n", " ")[:400]
        print(json.dumps(message, ensure_ascii=False), flush=True)
