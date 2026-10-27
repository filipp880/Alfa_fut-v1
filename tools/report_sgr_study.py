"""Независимый host-отчёт завершённого SGR-теста; без API и обучения.

Запуск: .tools/python312/python.exe tools/report_sgr_study.py runs/sgr/<id>
Читает только выбранный run, его выпускные reference-case и исходный TaskSpec.
Метрики пересчитываются sklearn, а не score/evaluate исследовательского стенда.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error

from contracts import ExperimentSpec
from data_io import encode_target, profile_data
from sgr_planner import make_sgr_schema, validate_sgr_response
from task_reader import load_task
from validation import make_splits

GRID = (0., .25, .5, .75, 1.)
GATEWAY = "https://api.alfahack-gateway26.ru/v1"
MODEL = "qwen/qwen3.8-27b"
LLM_CONFIG_FIELDS = {"model", "gateway_host", "gateway_path", "gateway_valid", "key_present", "max_calls",
    "token_limit", "timeout_seconds", "max_retries", "max_completion_tokens", "thinking_disabled",
    "invalid_token_limit_defaulted"}
LLM_SKIP_REASONS = {"disabled", "missing_key", "call_limit", "token_limit", "error_limit",
                    "search_time_limit", "context_limit", "token_reserve"}
LLM_ERROR_STAGES = {"import_client", "prepare", "client_init", "request", "response", "parse"}
LLM_FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter", "function_call", "unknown"}
LLM_EVENT_FIELDS = {"llm_usage": {"call", "tokens", "total_tokens", "completion_tokens", "finish_reason",
                                 "content_characters", "usage_source"},
    "llm_error": {"error_type", "http_status", "call", "stage", "finish_reason"},
    "llm_hypothesis": {"config"}, "llm_invalid_config": {"error_type"}, "sgr_summary": {"summary"},
    "sgr_invalid_response": {"error_type"}, "llm_context_limit": {"reason"},
    "llm_config": LLM_CONFIG_FIELDS, "llm_skipped": {"reason", "calls", "tokens", "errors"},
    "llm_invalid_metric": {"reason"}}


class AuditError(ValueError):
    """Недостающий или противоречивый артефакт; без вывода его содержимого."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def contained(root, value):
    """Никаких symlink, traversal или ссылок за пределы указанного каталога."""
    root = Path(root).resolve()
    path = root / value
    require(path.resolve().is_relative_to(root), "Путь артефакта покидает разрешённый каталог")
    for item in (path, *path.parents):
        require(not item.is_symlink(), "Symlink артефакта запрещён")
        if item == root:
            break
    return path.resolve()


def same_number(actual, expected, message):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10), message)


def checked_prediction(task, prediction, rows):
    p = np.asarray(prediction)
    expected = (rows, len(task.classes)) if task.classification else (rows,)
    require(p.shape == expected and p.dtype.kind in "fiu" and np.isfinite(p).all(),
            "Неверная форма, ось классов или неконечные predictions")
    if task.classification:
        require(not (p < 0).any() and not (p > 1).any()
                and np.allclose(p.sum(axis=1), 1., rtol=0., atol=1e-7),
                "Невалидные вероятности на замороженной оси классов")
    return p.astype(float, copy=False)


def independent_score(task, truth, prediction):
    """Исходная шкала регрессии; все классы TaskSpec, включая отсутствующие."""
    y = np.asarray(truth)
    require(y.ndim == 1 and len(y) > 0, "Пустая или неодномерная оценочная выборка")
    p = checked_prediction(task, prediction, len(y))
    if task.classification:
        require(task.metric == "macro_f1", "Этот протокол требует Macro F1")
        classes = np.arange(len(task.classes))
        require(y.dtype.kind in "iu" and np.isin(y, classes).all(), "Неверная ось истинных классов")
        labels = p.argmax(axis=1)
        return float(f1_score(y, labels, labels=classes, average="macro", zero_division=0))
    require(task.metric == "rmse" and np.isfinite(y).all(), "Этот протокол требует RMSE на исходной шкале")
    value = float(math.sqrt(mean_squared_error(y, p)))
    require(math.isfinite(value), "RMSE не является конечным числом")
    return value


def metric_details(task, truth, prediction):
    result = {"score": independent_score(task, truth, prediction)}
    if task.classification:
        labels = np.arange(len(task.classes))
        result.update(classes=task.classes, confusion_matrix=confusion_matrix(
            truth, np.asarray(prediction).argmax(axis=1), labels=labels).tolist(),
            per_class_f1=f1_score(truth, np.asarray(prediction).argmax(axis=1), labels=labels,
                                 average=None, zero_division=0).tolist())
    return result


def improves(task, value, previous):
    return (value - previous if task.classification else previous - value) > .001 * max(1., abs(previous))


def select_weight(task, truth, control, candidate):
    """Воспроизвести заранее заданный последовательный поиск с допуском."""
    scores = {str(weight): independent_score(task, truth, (1-weight)*control + weight*candidate)
              for weight in GRID}
    selected, best = 0., scores["0.0"]
    for weight in GRID[1:]:
        value = scores[str(weight)]
        if improves(task, value, best):
            selected, best = weight, value
    return selected, best, scores


def indices(value, rows, allow_empty=False):
    value = np.asarray(value)
    require(value.ndim == 1 and value.dtype.kind in "iu" and len(np.unique(value)) == len(value),
            "Индексы должны быть уникальными одномерными целыми")
    require((allow_empty or len(value)) and (not len(value) or (value.min() >= 0 and value.max() < rows)),
            "Пустые индексы или индекс вне train")
    return value


def equal_indices(actual, expected, message):
    require(np.array_equal(actual, expected), message)


def audit_split(path, rows, expected=None, full=False):
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == {"train", "valid"}, "Неверные поля split NPZ")
        train = indices(data["train"].copy(), rows)
        valid = indices(data["valid"].copy(), rows, allow_empty=full)
    require(not np.intersect1d(train, valid).size, "Train и validation пересекаются")
    if full:
        equal_indices(train, np.arange(rows), "Full не покрывает весь исходный train")
        require(not len(valid), "Full содержит validation")
    if expected is not None:
        equal_indices(train, expected.train, "Train не совпадает с исходным фиксированным split")
        equal_indices(valid, expected.valid, "Validation не совпадает с исходным фиксированным split")
    return train, valid


def source_schema_strict(source):
    """Статическая проверка SDK-обёртки в выпущенном коде, без его исполнения."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id == "options" and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "response_format"):
                fields = {key.value: value for key, value in zip(node.value.keys, node.value.values)
                          if isinstance(key, ast.Constant)}
                kind, inner = fields.get("type"), fields.get("json_schema")
                if not (isinstance(kind, ast.Constant) and kind.value == "json_schema" and isinstance(inner, ast.Dict)):
                    continue
                fields = {key.value: value for key, value in zip(inner.keys, inner.values)
                          if isinstance(key, ast.Constant)}
                strict, schema = fields.get("strict"), fields.get("schema")
                if (isinstance(strict, ast.Constant) and strict.value is True
                        and isinstance(schema, ast.Name) and schema.id == "response_schema"):
                    return True
    return False


def source_system_prompt(source, class_name):
    """Получить неизменяемую строку system из propose выпущенного планировщика."""
    tree = ast.parse(source)
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == class_name:
            for method in cls.body:
                if isinstance(method, ast.FunctionDef) and method.name == "propose":
                    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)
                             and isinstance(node.func, ast.Attribute) and node.func.attr == "request"
                             and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"]
                    require(len(calls) == 1 and bool(calls[0].args), "Неоднозначный system prompt в выпускном исходнике")
                    value = ast.literal_eval(calls[0].args[0])
                    require(type(value) is str, "System prompt не является фиксированной строкой")
                    return value
    raise AuditError("Нет propose/system prompt в выпускном планировщике")


def normalized_bind(value):
    value = str(value).replace("\\", "/").rstrip("/").lower()
    for prefix in ("/host_mnt/", "/run/desktop/mnt/host/", "/mnt/host/"):
        if value.startswith(prefix):
            tail = value[len(prefix):]
            if len(tail) > 1 and tail[1] == "/":
                value = tail[0] + ":" + tail[1:]
    return value


def audit_isolation(audit, case, protected):
    require(audit.get("status") == "verified" and audit.get("attested_before_fit") is True,
            "Нет предобучающего подтверждения изоляции")
    mounts = audit.get("actual_bind_mounts", [])
    require(len(mounts) == 1 and mounts[0].get("destination") == "/task"
            and mounts[0].get("rw") is True
            and normalized_bind(mounts[0].get("source")) == normalized_bind(case),
            "Контейнер не ограничен одним bind собственного case")
    require(all(not path.resolve().is_relative_to(case.resolve()) for path in protected),
            "Контрольные артефакты доступны внутри case")
    require(audit.get("network") == "none" and audit.get("llm_calls") == 0
            and audit.get("image_checkout_hidden") is True
            and audit.get("no_socket_or_home_mounts") is True and audit.get("readonly_root") is True,
            "Изоляция допускает сеть, чужие исходники, домашний каталог или API")
    require(0 < audit.get("cpu_total", 0) <= 8 and 0 < audit.get("memory_bytes_total", 0) <= 32_000_000_000,
            "Ресурсы выходят за пределы протокола")


def audit_reference(run, task_name):
    metadata_path = run / ("reference_" + task_name + ".json")
    prediction_path = run / ("reference_" + task_name + ".npz")
    metadata = read_json(metadata_path)
    case = contained(ROOT, metadata["release_case"])
    require(not case.is_relative_to(run), "Контроль должен происходить из отдельного выпускного case")
    release = read_json(case / "release_result.json")
    release_manifest = read_json(case / "release_manifest.json")
    archive_path = case / "source.zip"
    digest = sha(archive_path)
    require(digest == metadata["archive_sha256"] == release["archive_sha256"] == release_manifest["archive_sha256"],
            "SHA выпускного ZIP не совпадает с reference/выпуском")
    require(release.get("status") == "completed" and release.get("refit_completed") is True
            and not release.get("fallback_only"), "Reference не прошёл полный выпускной refit")
    summaries = sorted(case.glob(".alfa_runs/*/summary.json"))
    require(bool(summaries), "Нет основного выпускного summary")
    summary_path = summaries[-1]
    main = read_json(summary_path)
    require(main.get("status") == "completed" and main.get("refit_completed") is True
            and not main.get("fallback_only"), "Основной агент не завершил полный refit")
    require(main["selected_plan"] == metadata["plan"], "Контрольный план изменён после фиксации reference")
    sources = {}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and archive.testzip() is None, "Повреждённый или неоднозначный ZIP")
        for name in names:
            require(Path(name).name == name and "/" not in name and "\\" not in name
                    and (name.endswith(".py") or name == "requirements.txt"), "Небезопасный выпускной ZIP")
            require(name != Path(__file__).name, "Host-отчёт попал в runtime ZIP")
            contents = archive.read(name)
            sources[name] = hashlib.sha256(contents).hexdigest()
            require(sha(case / name) == sources[name], "Основные исходники изменены после выпускного ZIP")
        require(source_schema_strict(archive.read("llm_client.py").decode("utf-8-sig")),
                "В проверенном LLM-клиенте нет strict JSON Schema обёртки")
        system_prompts = {"normal": source_system_prompt(archive.read("llm_client.py").decode("utf-8-sig"), "LLMPlanner"),
                          "sgr": source_system_prompt(archive.read("sgr_planner.py").decode("utf-8-sig"), "SGRPlanner")}
    instrumentation = metadata.get("instrumentation_sha256", {})
    require(set(instrumentation) <= {"isolated_trial.py"}, "Незарегистрированное instrumentation-исключение")
    require(not (set(instrumentation) & set(sources)), "Instrumentation дублирует выпускные исходники")
    sources.update(instrumentation)
    source_dir = run / "sources" / task_name
    require({p.name for p in source_dir.iterdir()} == set(sources), "Снимок исходников содержит пропуски/посторонние файлы")
    for name, digest in sources.items():
        require(sha(contained(source_dir, name)) == digest, "Снимок исходников не совпадает с ZIP/instrumentation")
    task, train, test, _, dictionary = load_task(case / "task.md")
    require(task.classification == (task_name == "classification"), "Reference не соответствует задаче")
    require(sha(task.train_path) == metadata["train_sha256"] and sha(task.test_path) == metadata["test_sha256"],
            "Изменились исходные данные reference")
    y = encode_target(task, train)
    _, groups, times = profile_data(task, train, test, dictionary)
    splits = make_splits(task, y, groups, times)
    require(len(splits) >= 2, "Нет отдельного confirm")
    search, confirm = splits[:2]
    require(not np.intersect1d(np.r_[search.train, search.valid], confirm.valid).size,
            "Confirm участвовал в поисковых данных")
    for split in (search, confirm):
        if task.classification:
            require(not np.intersect1d(groups[split.train], groups[split.valid]).size,
                    "Группы одинаковых X пересекают train/validation")
        else:
            require(times is not None and np.max(times[split.train]) < np.min(times[split.valid]),
                    "Нарушена хронология регрессии")
    plan = metadata["plan"]
    members, weights = plan["members"], np.asarray(plan["weights"], dtype=float)
    require(bool(members) and len(set(members)) == len(members) and weights.shape == (len(members),)
            and np.isfinite(weights).all() and (weights >= 0).all() and np.isclose(weights.sum(), 1.),
            "Невалидный контрольный ансамбль")
    jobs = {}
    for path in summary_path.parent.glob("*/controller_result.json"):
        record = read_json(path)
        if record.get("status") == "ok" and record.get("config_key") in members:
            key = (record.get("phase"), record["config_key"])
            require(key not in jobs, "Неоднозначный основной worker job")
            jobs[key] = path.parent
    reconstructed = {}
    for phase, split in (("search", search), ("confirm", confirm), ("refit", None)):
        pieces = []
        for key in members:
            directory = jobs[(phase, key)]
            with np.load(directory / "predictions.npz", allow_pickle=False) as data:
                va = indices(data["valid_indices"], len(train), allow_empty=phase == "refit")
                equal_indices(va, split.valid if split else np.empty(0, dtype=int), "Reference indices не совпадают с split")
                valid = checked_prediction(task, data["valid"], len(va))
                test_p = checked_prediction(task, data["test"], len(test))
                pieces.append(valid.copy() if split else test_p.copy())
        reconstructed[phase] = sum(weight*piece for weight, piece in zip(weights, pieces))
    with np.load(prediction_path, allow_pickle=False) as data:
        require(set(data.files) == {"search", "confirm", "full_test", "search_indices", "confirm_indices"},
                "Reference NPZ содержит незаявленные поля")
        for phase, split in (("search", search), ("confirm", confirm)):
            equal_indices(data[phase + "_indices"], split.valid, "Reference NPZ имеет неверные индексы")
            require(np.array_equal(data[phase], reconstructed[phase]), "Reference predictions изменены")
        require(np.array_equal(data["full_test"], reconstructed["refit"]), "Reference full_test изменён")
    control = {phase: independent_score(task, y[split.valid], reconstructed[phase])
               for phase, split in (("search", search), ("confirm", confirm))}
    for phase in ("search", "confirm"):
        same_number(metadata[phase + "_score"], control[phase], "Метрика reference не воспроизводится")
    profile, _, _ = profile_data(task, train.iloc[search.train], test, dictionary)
    profile["time_column"] = read_json(summary_path.parent / "profile.json")["time_column"]
    audit_isolation(read_json(case / "resource_audit.json"), case, [])
    protected = [metadata_path, prediction_path, archive_path, summary_path]
    return dict(task=task, train=train, test=test, dictionary=dictionary, y=y, groups=groups, times=times, splits=(search, confirm),
                predictions=reconstructed, sources=sources, metadata=metadata, profile=profile, system_prompts=system_prompts,
                control=control, protected=protected,
                frozen_paths=[*protected, case / "release_result.json", *source_dir.iterdir()],
                evidence={"archive_sha256": sha(archive_path), "reference_npz_sha256": sha(prediction_path),
                          "source_sha256": sources, "control_search": control["search"],
                          "control_confirm": control["confirm"], "train_rows": len(train), "test_rows": len(test),
                          "search_rows": len(search.valid), "confirm_rows": len(confirm.valid),
                          "classes": task.classes, "metric": task.metric})


def audit_llm_diagnostics(event, arm):
    """Закрытый диагностический контракт; эти поля не используются как ML-метрики."""
    require(isinstance(event, dict) and type(event.get("event")) is str and event["event"] in LLM_EVENT_FIELDS,
            "Неизвестное событие журнала LLM")
    name = event["event"]
    require(set(event) <= LLM_EVENT_FIELDS[name] | {"time", "event"}, "Неизвестные поля журнала LLM")
    if name == "llm_config":
        require(LLM_CONFIG_FIELDS <= set(event), "Неполная диагностика конфигурации LLM")
        require(event["model"] == MODEL and event["gateway_host"] == "api.alfahack-gateway26.ru"
                and event["gateway_path"] == "/v1" and event["gateway_valid"] is True,
                "Диагностика LLM имеет другой model/gateway")
        require(type(event["key_present"]) is bool and (not arm["calls"] or event["key_present"])
                and type(event["invalid_token_limit_defaulted"]) is bool,
                "Неверные флаги конфигурации LLM")
        require(type(event["max_calls"]) is int and 0 <= event["max_calls"] <= 1
                and type(event["token_limit"]) is int and 1000 <= event["token_limit"] <= 40000
                and type(event["timeout_seconds"]) in (int, float) and event["timeout_seconds"] == 35
                and type(event["max_retries"]) is int and event["max_retries"] == 0
                and type(event["max_completion_tokens"]) is int and event["max_completion_tokens"] == 1000
                and event["thinking_disabled"] is True, "Диагностика LLM меняет бюджет/настройки парного теста")
    elif name == "llm_skipped":
        require({"reason", "calls", "tokens", "errors"} <= set(event)
                and type(event["reason"]) is str and event["reason"] in LLM_SKIP_REASONS,
                "Неизвестная причина пропуска LLM")
        require(type(event["calls"]) is int and 0 <= event["calls"] <= arm["calls"]
                and type(event["tokens"]) is int and 0 <= event["tokens"] <= arm["tokens"]
                and type(event["errors"]) is int and 0 <= event["errors"] <= 3,
                "Неверные счётчики пропуска LLM")
    elif name == "llm_error":
        require(type(event.get("error_type")) is str and event["error_type"].isidentifier()
                and len(event["error_type"]) <= 128, "Неверный тип ошибки LLM")
        status = event.get("http_status")
        require(status is None or (type(status) is int and 100 <= status <= 599), "Неверный HTTP status LLM")
        if "stage" in event:
            require(type(event["stage"]) is str and event["stage"] in LLM_ERROR_STAGES,
                    "Неизвестная стадия ошибки LLM")
    elif name == "llm_invalid_metric":
        require(event.get("reason") == "missing_or_invalid_metric", "Неизвестная причина invalid metric")
    if name in {"llm_usage", "llm_error"}:
        if "finish_reason" in event:
            require(event["finish_reason"] is None or (type(event["finish_reason"]) is str
                    and event["finish_reason"] in LLM_FINISH_REASONS),
                    "Неизвестный finish_reason LLM")
        if "usage_source" in event:
            require(type(event["usage_source"]) is str and event["usage_source"] in {"reported", "estimated"},
                    "Неизвестный источник token usage")


def prompt_audit(directory, arm, ref):
    prompt_path = directory / "prompt.json"
    prompt = read_json(prompt_path)
    allowed = {"system", "payload", "response_schema", "other_policy_context", "previous_reports_in_context", "request_settings"}
    require(set(prompt) <= allowed and allowed - {"request_settings"} <= set(prompt), "Неизвестные поля prompt")
    require(prompt["other_policy_context"] is False and prompt["previous_reports_in_context"] is False,
            "Prompt не подтверждает отсутствие других политик/предыдущих отчётов")
    require(prompt["system"] == ref["system_prompts"][arm["policy"]],
            "System prompt изменён относительно выпущенного планировщика")
    payload = prompt["payload"]
    require(set(payload) == {"task_excerpt", "task", "profile", "history", "remaining_seconds", "example"}
            and payload["history"] == [], "В prompt есть прошлая история или незаявленный контекст")
    task = ref["task"]
    require(payload["task"] == {"type": task.problem_type, "metric": task.metric,
            "features": task.feature_columns, "classes": task.classes}, "Prompt имеет чужой контракт задачи")
    require(payload["task_excerpt"] == Path(task.task_path).read_text(encoding="utf-8-sig")[:4500],
            "Task excerpt не совпадает с исходным условием")
    require(payload["profile"] == {**ref["profile"], "experiment_seed": arm["seed"]},
            "Prompt profile не соответствует только search-train")
    require(payload["example"] == ExperimentSpec().to_dict(), "Prompt example содержит дополнительный контекст")
    require(type(payload["remaining_seconds"]) is int and 0 <= payload["remaining_seconds"] <= 10800,
            "Неверный бюджет prompt")
    expected_schema = make_sgr_schema(task) if arm["policy"] == "sgr" else None
    require(prompt["response_schema"] == expected_schema, "Schema не соответствует политике или TaskSpec")
    settings = prompt.get("request_settings")
    strict_recorded = settings is not None
    if settings is not None:
        response_format = {"type": "json_schema", "json_schema": {"name": "ml_research_step",
            "strict": True, "schema": expected_schema}} if expected_schema else None
        require(settings == {"response_format": response_format, "temperature": .15, "max_tokens": 1000,
                "max_retries": 0, "enable_thinking": False, "model": MODEL, "gateway": GATEWAY},
                "Request settings не соответствуют заранее заданному протоколу")
    require(all(path.stat().st_mtime_ns <= prompt_path.stat().st_mtime_ns for path in ref["frozen_paths"]),
            "Reference/источники не были зафиксированы до запросов")
    events_path = directory / "experiments.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()] if events_path.exists() else []
    for event in events:
        audit_llm_diagnostics(event, arm)
    usage = [event for event in events if event["event"] == "llm_usage"]
    errors = [event for event in events if event["event"] == "llm_error"]
    require(type(arm["calls"]) is int and arm["calls"] in (0, 1) and len(usage) <= 1
            and len(errors) <= 1 and len(usage) <= arm["calls"]
            and (not arm["calls"] or usage or errors), "Число запросов/usage не воспроизводится")
    estimate = (len(prompt["system"]) + len(json.dumps(payload, ensure_ascii=False, default=str))
                + (len(json.dumps(expected_schema, ensure_ascii=False)) if expected_schema is not None else 0)) // 2 + 1400
    accounting = {"reported_confirmed": 0, "estimated_after_error": 0,
                  "estimated_usage_fallback": 0, "reported_or_estimated_unknown": 0}
    tokens = 0
    for event in usage:
        value = event["tokens"]
        require(type(value) is int and value >= 0, "Неверный token usage")
        require(event.get("call") == 1 and event.get("total_tokens") == value, "Неверная нумерация usage")
        completion = event.get("completion_tokens")
        require(completion is None or (type(completion) is int and 0 <= completion <= 1000), "Превышен completion limit")
        source = event.get("usage_source")
        if source == "estimated":
            require(value == estimate, "Fallback token usage не совпадает с консервативной оценкой")
            category = "estimated_usage_fallback"
        elif source == "reported":
            category = "reported_confirmed"
        else:
            category = "reported_confirmed" if completion is not None else "reported_or_estimated_unknown"
        accounting[category] += value
        tokens += value
    if arm["calls"] and not usage:
        tokens = estimate
        accounting["estimated_after_error"] = estimate
    require(arm["tokens"] == tokens, "Сумма usage/консервативной оценки не совпала с arm")
    hypotheses = [event["config"] for event in events if event["event"] == "llm_hypothesis"]
    require(len(hypotheses) == int(bool(arm["valid_proposal"])), "Нет однозначного валидированного предложения")
    schema_valid = None
    if arm["valid_proposal"]:
        config = ExperimentSpec.from_dict(arm["original_proposal"], task).to_dict()
        require(config == arm["original_proposal"] == hypotheses[0], "Исходное предложение изменено")
        if arm["policy"] == "sgr":
            envelopes = [event["summary"] for event in events if event["event"] == "sgr_summary"]
            require(len(envelopes) == 1, "Нет единственного SGR envelope")
            require(validate_sgr_response(envelopes[0], task).experiment.to_dict() == config,
                    "SGR envelope не соответствует предложению")
            schema_valid = True
        executed = {**config, "seed": arm["seed"]}
        require(arm["executed_config"] == executed, "Seed/DSL изменены кроме нормализации seed42/43")
    else:
        if arm["policy"] == "sgr":
            schema_valid = False
    return {"schema_valid": schema_valid, "strict_request_settings_recorded": strict_recorded,
            "tokens": tokens, "usage": accounting, "conservative_request_estimate": estimate,
            "prompt_sha256": sha(prompt_path), "journal_sha256": sha(events_path) if events_path.exists() else None}


def independent_csv(task, test, prediction, lineterminator="\n"):
    """Сериализовать CSV независимо от submission.make_submission/write_submission."""
    p = checked_prediction(task, prediction, len(test))
    result = test[task.id_columns].copy().reset_index(drop=True)
    columns = [name for name in task.output_columns if name not in task.id_columns]
    if task.classification:
        p = p / p.sum(axis=1, keepdims=True)
        if task.prediction_kind == "label":
            require(len(columns) == 1, "Неверный label CSV контракт")
            result[columns[0]] = np.asarray(task.classes)[p.argmax(axis=1)]
        elif task.prediction_kind == "probability":
            require(len(columns) == 1 and len(task.classes) == 2, "Неверный binary CSV контракт")
            result[columns[0]] = p[:, task.positive_index]
        elif task.prediction_kind == "probabilities":
            mapping = task.probability_indices or [task.classes.index(name) for name in task.probability_columns]
            require(sorted(mapping) == list(range(len(task.classes))) and columns == task.probability_columns,
                    "CSV не содержит полную ось классов")
            for name, index in zip(columns, mapping):
                result[name] = p[:, index]
        else:
            raise AuditError("Неизвестный классификационный CSV контракт")
    else:
        require(len(columns) == 1, "Неверный регрессионный CSV контракт")
        result[columns[0]] = p
    return result[task.output_columns].to_csv(index=False, lineterminator=lineterminator).encode("utf-8")


def audit_csv(task, test, prediction, path, recorded):
    expected = independent_csv(task, test, prediction)
    actual = Path(path).read_bytes()
    frame = pd.read_csv(BytesIO(actual), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    require(frame.columns.tolist() == task.output_columns and len(frame) == len(test), "CSV имеет неверные столбцы/размер")
    for column in task.id_columns:
        require(frame[column].tolist() == test[column].astype(str).tolist(), "CSV изменил ID или их порядок")
    require(not task.id_columns or not frame.duplicated(task.id_columns).any(), "CSV содержит повторные ID")
    digest = hashlib.sha256(actual).hexdigest()
    require(actual in (expected, independent_csv(task, test, prediction, "\r\n")),
            "CSV не воспроизводится из final full predictions и control full_test")
    require(recorded.get("status") == "ok" and recorded.get("sha256") == digest, "Записанный CSV audit SHA не совпадает")
    return {"status": "verified", "sha256": digest, "rows": len(frame), "columns": task.output_columns,
            "independently_rebuilt": True, "class_axis_and_finite": True}


def audit_arm(run, arm, ref, result=None):
    label = f"{arm['task']}_{arm['policy']}_{arm['seed']}"
    require(arm.get("label") == label and arm.get("source_reports_available") is False,
            "Arm label или защита чужих отчётов не соответствует протоколу")
    require(type(arm.get("valid_proposal")) is bool and type(arm.get("tokens")) is int and arm["tokens"] >= 0,
            "Неверные типы proposal/tokens")
    directory = contained(run / "prompts", label)
    if result is None:
        result = {name: arm.get(name) for name in ("label", "task", "policy", "seed", "status", "calls", "tokens", "valid_proposal")}
    result.update(prompt_audit(directory, arm, ref))
    result["ml_completed"] = False
    if not arm["valid_proposal"]:
        require(arm["status"] == "no_valid_proposal", "Невалидное предложение помечено завершённым ML")
        return result
    signature = {key: value for key, value in arm["executed_config"].items() if key not in {"seed", "hypothesis"}}
    result["proposal_recipe_sha256"] = hashlib.sha256(canonical(signature).encode("utf-8")).hexdigest()
    result["hypothesis_sha256"] = hashlib.sha256(arm["original_proposal"]["hypothesis"].strip().casefold().encode("utf-8")).hexdigest()
    result["model"] = arm["executed_config"]["model"]
    case = contained(run / "trials", label)
    if not case.exists() and arm.get("status") == "ml_failed":
        result.update(completed_phases=[], error_type=arm.get("error_type"),
                      ml_artifact_status="case_not_created")
        return result
    manifest_path = case / "manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("label") == label and manifest.get("original_train_mounted") is False
            and manifest.get("llm_calls") == 0, "Неверный manifest/изоляция case")
    require(set(manifest["source_files"]) == set(ref["sources"]), "ML source files не совпадают с ZIP+wrapper")
    for name, record in manifest["immutable_files"].items():
        path = contained(case, name)
        require(path.is_file() and sha(path) == record["sha256"] and path.stat().st_size == record["bytes"],
                "Изменён immutable файл case")
    for name, digest in ref["sources"].items():
        require(sha(contained(case, name)) == digest
                and manifest["immutable_files"][name]["sha256"] == digest, "ML source hash не совпадает с release snapshot")
    require(read_json(case / "config.json") == arm["executed_config"], "Case config отличается от executed config")
    original = ref["task"].to_dict()
    contract = read_json(case / "task_contract.json")
    for name, filename in (("task_path", "task.md"), ("train_path", "train.csv"), ("test_path", "test.csv"),
                           ("sample_path", "sample_submission.csv"), ("dictionary_path", "data_dictionary.csv")):
        original[name] = "/task/" + filename
    require(contract == original, "TaskSpec изменён внутри candidate case")
    for name in ("task.md", "train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv"):
        source = Path(ref["task"].task_path).parent / name
        record = manifest["original_input_hashes"][name]
        require(record["sha256"] == sha(source) and record["bytes"] == source.stat().st_size,
                "Candidate использовал другой исходный датасет")
    split_manifest = read_json(case / "split_manifest.json")
    require(split_manifest.get("target_exported") is False and split_manifest.get("contract_classes_frozen") is True
            and split_manifest.get("seed") == 42, "Split не зафиксирован до маскирования")
    task, rows = ref["task"], len(ref["train"])
    main_phase = "search_main" if task.classification else "search_late"
    main_split, confirm_split = ref["splits"]
    split_arrays = {}
    for phase, expected in ((main_phase, main_split), ("confirm", confirm_split), ("full", None)):
        tr, va = audit_split(case / "splits" / (phase + ".npz"), rows, expected, phase == "full")
        split_arrays[phase] = (tr, va)
    # Все зарегистрированные search исключают confirm: даже неисполненная фаза.
    for path in (case / "splits").glob("search_*.npz"):
        tr, va = audit_split(path, rows)
        require(not np.intersect1d(np.r_[tr, va], confirm_split.valid).size, "Confirm присутствует в одном из search split")
        if task.classification:
            require(not np.intersect1d(ref["groups"][tr], ref["groups"][va]).size, "Search разделяет одинаковые X")
        else:
            require(np.max(ref["times"][tr]) < np.min(ref["times"][va]), "Нарушена search-хронология")
    jobs = arm.get("jobs", [])
    require(len({job["phase"] for job in jobs}) == len(jobs), "Повторные ML jobs")
    cache = {}
    for job in jobs:
        phase = job["phase"]
        require(phase in (main_phase, "confirm", "full"), "Незарегистрированная фаза arm")
        phase_dir = contained(run, job["directory"])
        require(phase_dir == case / "phases" / phase, "Job находится вне собственного case")
        phase_summary = read_json(phase_dir / "summary.json")
        require(phase_summary.get("status") == "completed" and phase_summary.get("config") == arm["executed_config"],
                "Фаза не завершилась или изменила config")
        require(phase_summary.get("source_manifest_sha256") == sha(manifest_path), "Фаза использовала другой source manifest")
        audit = read_json(phase_dir / "runtime_isolation_attestation.json")
        audit_isolation(audit, case, ref["protected"])
        require(audit.get("manifest_sha256") == sha(manifest_path), "Attestation относится к другому manifest")
        host_request = read_json(phase_dir / "host_request.json")
        require(set(host_request) <= {"phase", "image_id", "container", "time_limit_seconds", "trial_seconds",
                "fixed_iterations", "seed", "trusted_original_mounted"} and host_request.get("trusted_original_mounted") is False,
                "Host request имеет незаявленные поля или монтирует оригинал")
        require(host_request.get("seed") in (None, arm["seed"]) and host_request.get("phase") == phase,
                "Host request изменил seed или фазу")
        tr, va = audit_split(phase_dir / "split.npz", rows, None, phase == "full")
        equal_indices(tr, split_arrays[phase][0], "Фаза изменила train split")
        equal_indices(va, split_arrays[phase][1], "Фаза изменила valid split")
        mask = read_json(phase_dir / "masking_audit.json")
        allowed = np.union1d(tr, va) if phase == main_phase else tr
        allowed_hash = hashlib.sha256(np.asarray(allowed, dtype="<i8").tobytes()).hexdigest()
        require(mask.get("allowed_label_indices_sha256") == allowed_hash and mask.get("allowed_label_rows") == len(allowed)
                and mask.get("masked_label_rows") == rows-len(allowed) and mask.get("original_train_mounted") is False
                and mask.get("original_train_sha256") == ref["metadata"]["train_sha256"]
                and mask.get("raw_feature_values_preserved") is True, "Неверная маска доступных меток")
        require(mask.get("validation_labels_available") is (phase == main_phase), "Фаза получила запрещённые validation labels")
        require(audit.get("masking_audit") == mask and phase_summary.get("label_blinded") is (phase == "confirm"),
                "Attestation/summary не соответствует действительной маске")
        request = read_json(phase_dir / "request.json")
        require(set(request) <= {"task", "config", "profile", "split_path", "threads", "refit", "fixed_iterations",
                "job_options", "fit_time_limit_seconds"}, "ML request имеет незаявленные поля")
        expected_profile, _, _ = profile_data(task, ref["train"].iloc[tr], ref["test"], ref["dictionary"])
        expected_profile["time_column"] = split_manifest["time_column"]
        require(request.get("task") == contract and request.get("config") == arm["executed_config"]
                and request.get("profile") == expected_profile and request.get("job_options") == {}
                and request.get("threads") == 8 and request.get("refit") is (phase != main_phase)
                and request.get("split_path") == f"/task/phases/{phase}/split.npz",
                "ML request содержит чужой контекст или изменённые task/config/profile")
        if phase == "confirm":
            require(mask.get("label_blinded") is True and phase_summary.get("label_blinded") is True
                    and phase_summary.get("score") is None and phase_summary.get("metrics", {}) == {}
                    and job.get("label_blinded") is True, "Confirm score/labels не скрыты от модели")
            worker = read_json(phase_dir / "result.json")
            require(worker.get("score") is None and worker.get("metrics", {}) == {}, "Confirm содержит dummy score")
        pred_path = phase_dir / "predictions.npz"
        require(sha(pred_path) == job["prediction_sha256"] == phase_summary["prediction_sha256"], "Prediction SHA не совпал")
        with np.load(pred_path, allow_pickle=False) as data:
            require(set(data.files) == {"valid", "test", "valid_indices"}, "Prediction NPZ содержит незаявленные поля")
            predicted_indices = indices(data["valid_indices"], rows, allow_empty=phase == "full")
            equal_indices(predicted_indices, va, "valid_indices не совпадают точно со split-файлом")
            cache[phase] = (checked_prediction(task, data["valid"], len(va)).copy(),
                            checked_prediction(task, data["test"], len(ref["test"])).copy(), phase_summary, phase_dir)
    if arm["status"] != "completed":
        require(arm["status"] == "ml_failed", "Неизвестный статус валидного ML предложения")
        result["completed_phases"] = list(cache)
        result["error_type"] = arm.get("error_type")
        return result
    require(list(cache) == [main_phase, "confirm", "full"], "Completed arm не имеет всех трёх полных фаз")
    frozen_path = directory / "frozen_before_confirm.json"
    frozen = read_json(frozen_path)
    require(frozen == arm["frozen"] and frozen["config"] == arm["executed_config"]
            and frozen["reference"] == ref["metadata"] and frozen["source_manifest_sha256"] == sha(manifest_path),
            "Замороженная конфигурация/reference/source manifest изменены")
    count = frozen["best_iterations"]
    require(type(count) is int and count > 0 and count == cache[main_phase][2]["best_iterations"],
            "Число итераций не взято из main search")
    require(all((cache[main_phase][3] / filename).stat().st_mtime_ns <= frozen_path.stat().st_mtime_ns
                for filename in ("predictions.npz", "summary.json")), "Freeze предшествует результатам main search")
    for phase in ("confirm", "full"):
        phase_dir = cache[phase][3]
        request = read_json(phase_dir / "request.json")
        require(set(request) <= {"task", "config", "profile", "split_path", "threads", "refit", "fixed_iterations",
                "job_options", "fit_time_limit_seconds"}, "ML request имеет незаявленные поля")
        require(request["config"] == frozen["config"] and request["fixed_iterations"] == count
                and request["refit"] is True, "Refit изменил frozen config/main count")
        for filename in ("host_request.json", "frozen_request.json", "summary.json"):
            record = read_json(phase_dir / filename)
            require(record.get("fixed_iterations") == count, "Confirm/full count отличается от frozen main count")
        require(all(frozen_path.stat().st_mtime_ns <= (phase_dir / filename).stat().st_mtime_ns
                    for filename in ("masking_audit.json", "host_request.json", "frozen_request.json")),
                "Freeze не предшествует confirm/full")
    y_search, y_confirm = ref["y"][main_split.valid], ref["y"][confirm_split.valid]
    alpha, blend_search, grid = select_weight(task, y_search, ref["predictions"]["search"], cache[main_phase][0])
    same_number(frozen["weight"], alpha, "Frozen alpha выбрана не по search grid")
    standalone_search = independent_score(task, y_search, cache[main_phase][0])
    same_number(frozen["search_score"], standalone_search, "Search standalone score не воспроизводится")
    same_number(frozen["search_blend_score"], blend_search, "Search blend score не воспроизводится")
    control = ref["predictions"]["confirm"]
    standalone = independent_score(task, y_confirm, cache["confirm"][0])
    frozen_p = (1-alpha)*control + alpha*cache["confirm"][0]
    blend = independent_score(task, y_confirm, frozen_p)
    effective = alpha if improves(task, blend, ref["control"]["confirm"]) else 0.
    same_number(arm["selected_weight"], effective, "Confirm изменил alpha иначе чем возвратом к контролю")
    for field, expected in (("standalone_confirm", standalone), ("frozen_blend_confirm", blend),
                            ("control_confirm", ref["control"]["confirm"]),
                            ("final_confirm", blend if effective else ref["control"]["confirm"])):
        same_number(arm[field], expected, "Confirm recorded score не воспроизводится: " + field)
    final = (1-effective)*ref["predictions"]["refit"] + effective*cache["full"][1]
    result.update(ml_completed=True, frozen_weight=alpha, effective_weight=effective, frozen_best_iterations=count,
                  search_grid=grid, search_standalone=standalone_search, search_frozen_blend=blend_search,
                  confirm_standalone=standalone, confirm_frozen_blend=blend,
                  confirm_effective=blend if effective else ref["control"]["confirm"],
                  confirm_details=metric_details(task, y_confirm, cache["confirm"][0]),
                  elapsed_seconds=arm.get("elapsed_seconds"),
                  phase_seconds={job["phase"]: job["seconds"] for job in jobs},
                  source_manifest_sha256=sha(manifest_path), frozen_sha256=sha(frozen_path),
                  submission=audit_csv(task, ref["test"], final, directory / "submission.csv", arm["submission_audit"]))
    return result


def paired_comparisons(arms):
    pairs = []
    for task in ("regression", "classification"):
        for seed in (42, 43):
            match = {arm["policy"]: arm for arm in arms if arm["task"] == task and arm["seed"] == seed}
            available = set(match) == {"normal", "sgr"} and all(arm.get("ml_completed") and not arm.get("audit_errors")
                                                                 for arm in match.values())
            row = {"task": task, "seed": seed, "same_holdout": True if available else None, "available": available}
            if available:
                direction = 1 if task == "classification" else -1
                for field in ("search_standalone", "search_frozen_blend", "confirm_standalone", "confirm_frozen_blend", "confirm_effective"):
                    row[field + "_delta_sgr_minus_normal"] = match["sgr"][field] - match["normal"][field]
                    row[field + "_improvement_sgr"] = direction*(match["sgr"][field] - match["normal"][field])
            pairs.append(row)
    return pairs


def build_report(run):
    run = Path(run).resolve()
    require(run.is_relative_to(ROOT / "runs" / "sgr"), "Run должен находиться в runs/sgr/<id>")
    summary = read_json(run / "summary.json")
    require(summary.get("status") == "completed", "SGR run ещё не завершён: отчёт результатов не создаётся")
    report = {"report_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
              "run": run.relative_to(ROOT).as_posix(), "summary_sha256": sha(run / "summary.json"),
              "status": "audit_failed", "study_completed": False, "audit_errors": [], "arms": [],
              "limitations": ["Два seed на одних holdout не являются новыми независимыми тестами; статистическая значимость не оценивается.",
                              "Принятый response_format и клиентская проверка не раскрывают внутреннюю реализацию constrained decoding сервера.",
                              "Старый llm_usage без usage_source/completion_tokens не различает reported total_tokens и estimate; он отмечен unknown.",
                              "Уникальность гипотез определяется точным нормализованным текстом и DSL, а не семантическим сравнением.",
                              "Проверяются сохранённые hash/mtime/attestation; это проверка артефактов, а не независимая телеметрия Docker/API."]}
    errors = report["audit_errors"]
    for condition, message in ((summary.get("gateway") == GATEWAY and summary.get("model") == MODEL, "Неверный шлюз/модель"),
            (summary.get("max_calls") == 8 and summary.get("completion_limit") == 1000, "Неверный предел запросов/completion"),
            (type(summary.get("token_limit")) is int and 1000 <= summary["token_limit"] <= 40000, "Неверный token limit"),
            (summary.get("protocol_sha256") == sha(ROOT / "docs/SGR_PROTOCOL.md"), "Протокол изменён после фиксации")):
        if not condition:
            errors.append(message)
    refs = {}
    report["references"] = {}
    for task in ("regression", "classification"):
        try:
            refs[task] = audit_reference(run, task)
            report["references"][task] = refs[task]["evidence"]
        except (OSError, ValueError, KeyError, TypeError, IndexError, SyntaxError, zipfile.BadZipFile) as error:
            errors.append("Reference " + task + ": " + (str(error) if isinstance(error, AuditError) else type(error).__name__))
    if len(refs) == 2:
        if not (refs["regression"]["metadata"]["archive_sha256"] == refs["classification"]["metadata"]["archive_sha256"]
                == summary.get("reference_archive_sha256")):
            errors.append("Две задачи используют разные reference ZIP")
    expected = [(task, seed, policy) for task in ("regression", "classification") for seed in (42, 43)
                for policy in (("normal", "sgr") if (task == "regression") == (seed == 42) else ("sgr", "normal"))]
    arms = summary.get("arms", [])
    actual = [(arm.get("task"), arm.get("seed"), arm.get("policy")) for arm in arms]
    if actual != expected[:len(actual)] or len(actual) > 8 or len(set(actual)) != len(actual):
        errors.append("Порядок/уникальность arms не соответствует парному протоколу")
    if len(actual) < 8 and summary.get("budget_exhausted") is not True:
        errors.append("Недостающие arms не объяснены исчерпанием бюджета")
    for arm in arms:
        result = {key: arm.get(key) for key in ("label", "task", "policy", "seed", "status", "calls", "tokens", "valid_proposal")}
        try:
            require(arm.get("task") in refs and arm.get("seed") in (42, 43) and arm.get("policy") in ("normal", "sgr"),
                    "Нет проверенного reference или недопустимый task/policy/seed")
            result = audit_arm(run, arm, refs[arm["task"]], result)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            result.update(ml_completed=False, audit_errors=[str(error) if isinstance(error, AuditError) else type(error).__name__])
        report["arms"].append(result)
    calls = sum(arm.get("calls", 0) for arm in arms)
    tokens = sum(arm.get("tokens", 0) for arm in arms)
    if not (calls == summary.get("calls") and 0 <= calls <= 8 and tokens == summary.get("tokens")
            and 0 <= tokens <= summary.get("token_limit", 0)):
        errors.append("Общий token/call бюджет не воспроизводится")
    report["accounting"] = {"calls": calls, "accounted_tokens": tokens, "token_limit": summary.get("token_limit"),
                            "budget_exhausted": summary.get("budget_exhausted", False),
                            "usage": {name: sum(arm.get("usage", {}).get(name, 0) for arm in report["arms"])
                                      for name in ("reported_confirmed", "estimated_after_error", "estimated_usage_fallback",
                                                   "reported_or_estimated_unknown")}}
    report["paired_comparisons"] = paired_comparisons(report["arms"])
    report["proposal_diversity"] = {"valid": sum(bool(arm.get("valid_proposal")) for arm in report["arms"]),
        "unique_recipe_hashes": len({arm["proposal_recipe_sha256"] for arm in report["arms"] if "proposal_recipe_sha256" in arm}),
        "unique_hypothesis_hashes": len({arm["hypothesis_sha256"] for arm in report["arms"] if "hypothesis_sha256" in arm})}
    audited = not errors and not any(arm.get("audit_errors") for arm in report["arms"])
    report["study_completed"] = audited and len(arms) == 8 and all(arm.get("ml_completed") for arm in report["arms"])
    report["status"] = "completed" if report["study_completed"] else "completed_with_failures" if audited else "audit_failed"
    return report


def markdown_report(report):
    def number(value):
        return "—" if value is None else f"{value:.8g}"
    lines = ["# Реальный парный SGR-тест", "", f"Run: `{report['run']}`. Статус независимой проверки: **{report['status']}**.", "",
             "RMSE пересчитан sklearn на исходной шкале; Macro F1 — по всем классам исходного TaskSpec. "
             "Для классификации в JSON сохранены confusion matrix и F1 каждого класса. "
             "Знак Δ в парной таблице равен SGR − обычный планировщик: для RMSE меньше лучше, для Macro F1 больше лучше.", "",
             "| Задача | Политика | Seed | Статус | Tokens | Schema | Search отдельно | Search frozen blend | Confirm отдельно | Confirm frozen blend | Confirm effective | α frozen → effective | Секунд | CSV |",
             "|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|---:|---|"]
    for arm in report["arms"]:
        status = "audit_failed" if arm.get("audit_errors") else arm.get("status", "unknown")
        schema = "да" if arm.get("schema_valid") is True else "нет" if arm.get("schema_valid") is False else "не применима"
        values = [arm.get("task"), arm.get("policy"), str(arm.get("seed")), status, str(arm.get("tokens", "—")), schema,
                  *[number(arm.get(field)) for field in ("search_standalone", "search_frozen_blend", "confirm_standalone",
                                                         "confirm_frozen_blend", "confirm_effective")],
                  number(arm.get("frozen_weight")) + " → " + number(arm.get("effective_weight")),
                  number(arm.get("elapsed_seconds")),
                  "проверен" if arm.get("submission", {}).get("status") == "verified" else "—"]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    lines.extend(["", "| Задача | Seed | Δ Confirm отдельно | Δ Confirm frozen blend | Δ Confirm effective |",
                  "|---|---:|---:|---:|---:|"])
    for pair in report["paired_comparisons"]:
        lines.append("| " + " | ".join([pair["task"], str(pair["seed"]), *[number(pair.get(field + "_delta_sgr_minus_normal"))
            for field in ("confirm_standalone", "confirm_frozen_blend", "confirm_effective")]]) + " |")
    accounting = report["accounting"]
    lines.extend(["", f"Запросов: {accounting['calls']}/8. Учтено токенов: {accounting['accounted_tokens']}/{accounting['token_limit']}. "
                  f"Подтверждённый reported usage: {accounting['usage']['reported_confirmed']}; "
                  f"estimate после ошибки: {accounting['usage']['estimated_after_error']}; "
                  f"estimate без reported usage: {accounting['usage'].get('estimated_usage_fallback', 0)}; "
                  f"reported/estimate неизвестно: {accounting['usage']['reported_or_estimated_unknown']}.", ""])
    for task, ref in report["references"].items():
        lines.append(f"Контроль {task}: search {number(ref['control_search'])}, confirm {number(ref['control_confirm'])}; "
                     f"ZIP SHA256 `{ref['archive_sha256']}`.")
    failures = report["audit_errors"] + [f"{arm['label']}: {message}" for arm in report["arms"] for message in arm.get("audit_errors", [])]
    if failures:
        lines.extend(["", "Проверка артефактов не пройдена:", "", *["- " + message for message in failures]])
    if not report["study_completed"]:
        lines.extend(["", "Полное ML-сравнение не завершено: отсутствующие/неуспешные API или ML arms не заменяются результатами."])
    lines.extend(["", "Ограничения:", "", *["- " + item for item in report["limitations"]], "",
                  "Два повтора описывают устойчивость одного предложения на тех же данных. "
                  "Они не доказывают общего превосходства SGR или всего многошагового AutoResearch. "
                  "Результаты не меняют контроль и выбор планировщика автоматически.", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="Завершённый runs/sgr/<id>")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs")
    args = parser.parse_args(argv)
    try:
        report = build_report(args.run)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("Отчёт не создан: " + (str(error) if isinstance(error, AuditError) else type(error).__name__), file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("SGR_RESULTS.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                          ("SGR_RESEARCH.md", markdown_report(report))):
        path = args.output_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        temporary.replace(path)
    print(json.dumps({"status": report["status"], "study_completed": report["study_completed"],
                      "arms": len(report["arms"]), "output": str(args.output_dir)}, ensure_ascii=False))
    return 0 if report["study_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
