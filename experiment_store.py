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
        message = {k: record[k] for k in ("event", "phase", "model", "score", "status", "reason") if k in record}
        print(json.dumps(message, ensure_ascii=False), flush=True)
