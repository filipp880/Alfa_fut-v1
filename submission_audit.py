"""Независимый аудит записанного CSV и безопасный манифест запуска."""
from __future__ import annotations

import hashlib
from io import BytesIO
import math
from pathlib import Path

import numpy as np
import pandas as pd

from contracts import TaskSpec
from experiment_store import atomic_json


def _numeric(frame: pd.DataFrame) -> np.ndarray:
    try:
        values = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Предсказания CSV должны быть числовыми") from None
    if not np.isfinite(values).all():
        raise ValueError("CSV содержит NaN или бесконечные предсказания")
    return values


def _statistics(values: np.ndarray) -> dict:
    if not values.size:
        return {"min": None, "max": None, "mean": None}
    # Деление перед суммированием не даёт переполниться среднему конечных чисел.
    flat = values.reshape(-1)
    return {"min": float(flat.min()), "max": float(flat.max()),
            "mean": float(math.fsum(float(v) / len(flat) for v in flat))}


def audit_submission(task: TaskSpec, test: pd.DataFrame, path: Path | str) -> dict:
    """Проверить уже сериализованный CSV; при нарушении контракта — ValueError.

    Хэш и проверка относятся к одному снимку байтов. ID и метки читаются как
    строки, поэтому ведущие нули и строковые классы вроде ``NA`` сохраняются.
    Допустимая абсолютная погрешность суммы вероятностей равна 1e-6.
    """
    payload = Path(path).read_bytes()
    try:
        frame = pd.read_csv(BytesIO(payload), dtype=str, keep_default_na=False,
                            encoding="utf-8-sig")
    except (ValueError, pd.errors.ParserError, UnicodeError):
        raise ValueError("Не удалось прочитать CSV ответа") from None
    expected = list(task.output_columns)
    if len(expected) != len(set(expected)) or frame.columns.tolist() != expected:
        raise ValueError("Столбцы CSV не совпадают с контрактом задачи")
    if len(frame) != len(test):
        raise ValueError("Число строк CSV не совпадает с test")
    if any(column not in test or column not in frame for column in task.id_columns):
        raise ValueError("В CSV или test отсутствует идентификатор")
    if task.id_columns:
        if frame.duplicated(task.id_columns).any():
            raise ValueError("CSV содержит повторные идентификаторы")
        for column in task.id_columns:
            if frame[column].tolist() != test[column].astype(str).tolist():
                raise ValueError("CSV изменяет порядок или значения идентификаторов")
    columns = [column for column in expected if column not in task.id_columns]
    result = {"status": "ok", "sha256": hashlib.sha256(payload).hexdigest(),
              "rows": len(frame), "columns": expected,
              "prediction_kind": task.prediction_kind}
    if task.classification and task.prediction_kind == "label":
        if len(columns) != 1:
            raise ValueError("Для меток ожидается один столбец предсказаний")
        classes = [str(value) for value in task.classes]
        labels = frame[columns[0]]
        if not labels.isin(classes).all():
            raise ValueError("CSV содержит недопустимые метки классов")
        counts = labels.value_counts()
        result["label_distribution"] = {label: int(counts.get(label, 0)) for label in classes}
    elif task.classification:
        if task.prediction_kind not in {"probability", "probabilities"}:
            raise ValueError("Неизвестный формат классификационного ответа")
        if task.prediction_kind == "probability" and len(columns) != 1:
            raise ValueError("Для вероятности ожидается один столбец")
        if task.prediction_kind == "probabilities" and (
                columns != list(task.probability_columns) or len(columns) != len(task.classes)):
            raise ValueError("Неверная схема столбцов вероятностей")
        values = _numeric(frame[columns])
        if ((values < 0) | (values > 1)).any():
            raise ValueError("Вероятности CSV находятся вне диапазона [0, 1]")
        if task.prediction_kind == "probabilities":
            sums = values.sum(axis=1)
            if not np.allclose(sums, 1., rtol=0., atol=1e-6):
                raise ValueError("Суммы вероятностей CSV не равны единице")
            result["probability_row_sums"] = _statistics(sums)
        result["prediction_summary"] = _statistics(values)
    else:
        if len(columns) != 1:
            raise ValueError("Для регрессии ожидается один столбец предсказаний")
        result["prediction_summary"] = _statistics(_numeric(frame[columns]))
    return result


_RUN_STATUSES = {
    "initializing", "completed", "deadline", "recovered_with_submission",
    "fallback_unsupported_metric", "fallback_small_or_constant_task",
    "fallback_no_successful_experiments", "failed", "refitting", "searching",
    "confirming", "repeating",
}
_BOOLEAN_FIELDS = {"fallback_only", "refit_completed", "final_refit_completed", "history_operator_eligible", "oof_selected", "oof_selected_on_search"}
_NUMBER_FIELDS = {
    "elapsed_seconds", "time_limit_seconds", "threads", "memory_limit_mb",
    "oof_fit_attempts", "oof_completed_fits", "oof_validated_fits", "oof_elapsed_seconds",
    "experiments", "successful_experiments", "search_best_score", "llm_calls",
    "llm_tokens", "peak_worker_rss_mb", "confirmation_score",
}


def _safe_run_summary(summary: dict) -> dict:
    """Allowlist вместо копирования summary, конфигураций, путей или окружения."""
    status = summary.get("status")
    result = {"status": status if isinstance(status, str) and status in _RUN_STATUSES else "unknown"}
    if summary.get("prediction_source") in {"fallback", "search", "confirm", "refit"}:
        result["prediction_source"] = summary["prediction_source"]
    if summary.get("oof_status") in {"disabled", "skipped", "completed", "discarded", "failed"}:
        result["oof_status"] = summary["oof_status"]
    for name in sorted(_BOOLEAN_FIELDS):
        value = summary.get(name)
        if isinstance(value, (bool, np.bool_)):
            result[name] = bool(value)
    for name in sorted(_NUMBER_FIELDS):
        value = summary.get(name)
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
            if math.isfinite(float(value)):
                result[name] = value.item() if hasattr(value, "item") else value
    return result


def write_run_manifest(directory: Path | str, task: TaskSpec, summary: dict) -> dict:
    """Аудировать итоговый CSV и атомарно записать два одинаковых манифеста.

    Артефакты: ``directory/manifest.json`` и ``<task_dir>/submission.manifest.json``.
    Ошибка аудита сохраняется как status=failed, не скрывая уже имеющийся CSV.
    summary читается только через allowlist; секреты и строки данных не копируются.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(task.task_path).resolve().parent
    location = summary.get("output", root / "submission.csv")
    path = Path(location) if isinstance(location, (str, Path)) else root / "submission.csv"
    if not path.is_absolute():
        path = root / path
    try:
        test = pd.read_csv(task.test_path, dtype={column: str for column in task.id_columns},
                           usecols=task.id_columns or [0], keep_default_na=False,
                           encoding="utf-8-sig")
        audit = audit_submission(task, test, path)
    except (OSError, ValueError, UnicodeError) as error:
        # Ни текст исключения, ни содержимое CSV не попадают в манифест.
        audit = {"status": "failed", "error_type": type(error).__name__,
                 "reason": "Итоговый CSV не прошёл проверку или недоступен"}
    sources = {}
    for source in sorted(Path(__file__).resolve().parent.glob("*.py")):
        if source.is_file():
            sources[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {"manifest_version": 1, "status": audit["status"],
                "run": _safe_run_summary(summary), "submission": audit,
                "source_sha256": sources}
    atomic_json(directory / "manifest.json", manifest)
    atomic_json(root / "submission.manifest.json", manifest)
    return manifest
