"""Исторический CatBoost с явным резервом ExtraTrees для неизвестных ключей.

История и все политики X обучаются только на переданных ``tr``. Обучающая
матрица CatBoost содержит ВСЕ tr-строки; наличие истории управляет только
выбором предсказания. Схема сущности задаётся метаданными, без доступа к y.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import math
import re
import time

import numpy as np
import pandas as pd

from contracts import ExperimentSpec, TaskSpec, fingerprint
from historical_features import (HISTORY_COLUMNS, HistoricalFeatures,
                                 HistorySchema, suggest_history_schema)
from models import train_model
from novel_models import AlternativeModel, NovelSpec


MIN_HISTORY_COVERAGE = .05
_SCHEMA_FIELDS = {"time_column", "entity_columns", "feature_columns",
                  "forbidden_columns", "min_count"}


def _task_schema(data, task):
    if not isinstance(data, dict) or set(data) - _SCHEMA_FIELDS:
        raise ValueError("Недопустимая схема исторического маршрутизатора")
    if "feature_columns" in data:
        supplied = data["feature_columns"]
        if not isinstance(supplied, (list, tuple)) or list(supplied) != list(task.feature_columns):
            raise ValueError("Схема истории должна использовать признаки текущей задачи")
    extra_forbidden = data.get("forbidden_columns", ())
    if not isinstance(extra_forbidden, (list, tuple)) or any(
            not isinstance(c, str) or not c for c in extra_forbidden):
        raise ValueError("Неверный список запрещённых столбцов истории")
    forbidden = tuple(dict.fromkeys([*task.id_columns, task.target, *extra_forbidden]))
    return HistorySchema(time_column=data.get("time_column"),
                         entity_columns=data.get("entity_columns", ()),
                         feature_columns=tuple(task.feature_columns),
                         forbidden_columns=forbidden,
                         min_count=data.get("min_count", 1)).validate()


@dataclass
class HistoryRouterSpec(ExperimentSpec):
    model: str = "history_router"
    iterations: int = 2400
    depth: int = 8
    learning_rate: float = .045
    history_schema: dict = field(default_factory=dict)
    history_policy: str = "native"
    fallback: str = "forest"
    fallback_weight: float = .5
    fallback_iterations: int = 2400
    seed_count: int = 1
    hypothesis: str = "Строго прошлая история и резервный ExtraTrees"

    def validate(self, task):
        if self.model != "history_router" or task.classification:
            raise ValueError("Исторический маршрутизатор требует регрессию")
        if task.target in task.feature_columns or set(task.id_columns) & set(task.feature_columns):
            raise ValueError("История не может использовать target или row ID")
        standard = {f.name: getattr(self, f.name) for f in fields(ExperimentSpec)}
        standard["model"] = "catboost"
        ExperimentSpec.from_dict(standard, task)
        _task_schema(self.history_schema, task)
        if self.history_policy not in {"native", "median", "missing_pattern"}:
            raise ValueError("Неизвестная политика исходных числовых признаков")
        if self.fallback not in {"forest", "forest_huber"}:
            raise ValueError("Неизвестная резервная ветвь")
        if (isinstance(self.fallback_weight, bool) or not isinstance(self.fallback_weight, (int, float))
                or not math.isfinite(self.fallback_weight) or not 0 <= self.fallback_weight <= 1):
            raise ValueError("Вес Huber должен быть в [0, 1]")
        if (isinstance(self.fallback_iterations, bool) or not isinstance(self.fallback_iterations, int)
                or not 50 <= self.fallback_iterations <= 6000):
            raise ValueError("Неверное фиксированное число Huber-итераций")
        if isinstance(self.seed_count, bool) or not isinstance(self.seed_count, int) or self.seed_count not in {1, 2}:
            raise ValueError("seed_count должен быть 1 или 2")
        if self.seed + self.seed_count - 1 >= 2**31:
            raise ValueError("Второй seed выходит за допустимый диапазон")
        return self

    @property
    def key(self):
        data = self.to_dict()
        for name in ("hypothesis", "leaves", "min_child_samples"):
            data.pop(name, None)
        return fingerprint(data)


class RouterNumericPolicy:
    """Две политики X без зависимости production от исследовательских рецептов."""
    def __init__(self, kind, columns):
        if kind not in {"median", "missing_pattern"}:
            raise ValueError("Неизвестная числовая политика")
        self.kind, self.columns = kind, list(columns)

    def _numeric(self, frame):
        return frame[self.columns].apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf], np.nan)

    def fit(self, frame):
        values = self._numeric(frame)
        if self.kind == "median":
            # Как в исходном G2: не добавляем постоянные индикаторы колонок,
            # в которых обучающая часть не содержит пропусков.
            self.columns = [c for c in self.columns if values[c].isna().any()]
            values = values[self.columns]
        self.medians_ = values.median().fillna(0.).to_dict()
        self.pattern_columns_ = [c for c in self.columns if values[c].isna().any()][:8]
        added = ([c + "__was_missing" for c in self.columns] if self.kind == "median"
                 else ["__row_missing_count", "__row_zero_count", "__missing_pattern"])
        if set(added) & set(frame.columns) or len(set(added)) != len(added):
            raise ValueError("Служебные столбцы числовой политики уже существуют")
        self.fit_rows_ = len(frame)
        return self

    def transform(self, frame):
        result, values = frame.copy(), self._numeric(frame)
        if self.kind == "median":
            for column in self.columns:
                result[column + "__was_missing"] = values[column].isna().astype(float)
                result[column] = values[column].fillna(self.medians_[column])
        else:
            result["__row_missing_count"] = values.isna().sum(axis=1).astype(float)
            result["__row_zero_count"] = values.eq(0).sum(axis=1).astype(float)
            pattern = np.zeros(len(frame), dtype=float)
            for i, column in enumerate(self.pattern_columns_):
                pattern += values[column].isna().to_numpy() * 2**i
            result["__missing_pattern"] = pattern
        return result


def _feature_diagnostics(history, frame):
    history.transform(frame)
    diagnostics = {name: history.diagnostics_[name] for name in (
        "fit_rows", "entity_keys", "missing_key_rows", "missing_time_rows",
        "multirow_key_period_groups", "multirow_key_period_rows")}
    diagnostics.update(history.last_diagnostics_)
    numeric_time = pd.to_numeric(frame[history.schema.time_column], errors="coerce").to_numpy(dtype=float)
    periods = len(np.unique(numeric_time[np.isfinite(numeric_time)]))
    diagnostics.update(period_count=periods, minimum_coverage=MIN_HISTORY_COVERAGE,
                       ambiguous_row_fraction=diagnostics["multirow_key_period_rows"] / max(1, len(frame)),
                       eligible=bool(periods >= 2 and diagnostics["coverage"] >= MIN_HISTORY_COVERAGE))
    return diagnostics


def history_schema_diagnostics(schema, frame):
    """Покрытие и неоднозначные группы по одним X, никогда по таргетам.

    Нулевые служебные метки нужны только для повторного использования механизма
    строгого прошлого. Значения реального таргета здесь не читаются.
    """
    if isinstance(schema, dict):
        schema = HistorySchema(**schema)
    history = HistoricalFeatures(schema).fit(frame, np.zeros(len(frame), dtype=float))
    return _feature_diagnostics(history, frame)


def suggest_router_schema(dictionary, task, profile, frame):
    """Предложить схему только по словарю и X поисковой обучающей части.

    Поддерживаются явно описанный ключ сущности (включая части составного
    ключа), явно описанные статические атрибуты и прежняя семантика зданий.
    Обычный row ID, корреляции с y и имена датасетов не используются.
    """
    time_column = profile.get("time_column")
    if (task.classification or not isinstance(dictionary, pd.DataFrame) or not isinstance(frame, pd.DataFrame)
            or not {"feature", "role", "description_ru"} <= set(dictionary.columns)
            or time_column not in task.feature_columns or time_column not in frame):
        return {}
    forbidden = set(task.id_columns + [task.target])
    allowed = set(task.feature_columns) - forbidden - {time_column}
    key_pattern = re.compile(
        r"(?:entity|object|asset|subject)[ _-]*(?:id|key|identifier)\b|"
        r"(?:идентификатор|код|ключ)\s+(?:повторно\s+измеряемого\s+)?(?:объекта|сущности|актива|пациента)|"
        r"част[ьи]\s+составного\s+ключа", re.I)
    static_pattern = re.compile(
        r"\b(?:static|stable|invariant)\s+(?:(?:entity|object)\s+)?(?:attribute|feature)\b|"
        r"(?:статическ\w*|неизменн\w*|постоянн\w*)\s+(?:атрибут|признак|характеристик)", re.I)
    row_pattern = re.compile(
        r"\b(?:row|record|observation)[ _-]*(?:id|identifier|number)\b|"
        r"(?:идентификатор|номер|ключ)\s+(?:строки|записи|наблюдения)", re.I)
    key_rows, static = [], []
    seen = set()
    for row in dictionary.to_dict("records"):
        column, role = row["feature"], str(row["role"]).strip().lower()
        description = str(row.get("description_ru", ""))
        if column not in allowed or column in seen or row_pattern.search(description):
            continue
        if role not in {"feature", "entity_key", "entity_id", "entity_attribute", "static_attribute"}:
            continue
        seen.add(column)
        if role in {"entity_key", "entity_id"} or key_pattern.search(description):
            key_rows.append((column, description, role))
        elif role in {"entity_attribute", "static_attribute"} or static_pattern.search(description):
            static.append(column)
    explicit = [row[0] for row in key_rows]
    # Несколько независимых ID не превращаются молча в составной ключ.
    if len(explicit) > 1 and not all(
            row[2] == "entity_key" or re.search(r"composite|составн", row[1], re.I)
            for row in key_rows):
        return {}
    building = suggest_history_schema(dictionary, task.feature_columns, time_column,
                                      task.id_columns + [task.target])
    columns = explicit or building.get("entity_columns", []) or static
    if not 1 <= len(columns) <= 8:
        return {}
    try:
        schema = _task_schema({"time_column": time_column, "entity_columns": columns}, task)
        diagnostics = history_schema_diagnostics(schema, frame[task.feature_columns])
    except (ValueError, TypeError, KeyError):
        return {}
    if not diagnostics["eligible"]:
        return {}
    return {"time_column": schema.time_column, "entity_columns": list(schema.entity_columns),
            "feature_columns": list(schema.feature_columns),
            "forbidden_columns": list(schema.forbidden_columns), "min_count": schema.min_count}


def _augment(frame, history, policy):
    # Сопоставление выполняется по исходному ключу, до медианной подстановки.
    historical = history.transform(frame)
    transformed = policy.transform(frame) if policy is not None else frame.copy()
    for column in HISTORY_COLUMNS:
        transformed[column] = historical[column].to_numpy()
    return transformed


def _predict_history(bundle, frame):
    if not len(frame):
        return np.empty(0, dtype=float)
    prediction = np.asarray(bundle["model"].predict(bundle["builder"].transform(frame)), dtype=float).reshape(-1)
    if bundle["config"].get("target_transform") == "log1p":
        prediction = np.expm1(np.clip(prediction, -50, 50))
    if bundle["task"]["metric"] == "rmsle":
        prediction = np.maximum(prediction, 0)
    return prediction


def predict_router(bundle, frame):
    """Предсказание по сохранённой истории, политике и всем seed-компонентам."""
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Ожидается DataFrame с уникальными столбцами")
    original = frame[bundle["task"]["feature_columns"]]
    if not len(original):
        return np.empty(0, dtype=float)
    augmented = _augment(original, bundle["history"], bundle["policy"])
    known = augmented["__history_known"].to_numpy(dtype=bool)
    history_models, forest_models = bundle["history_models"], bundle["forest_models"]
    if not history_models or len(history_models) != len(forest_models):
        raise ValueError("Неполный состав исторического маршрутизатора")
    predictions = []
    for number, (history_model, forest_model) in enumerate(zip(history_models, forest_models)):
        base = forest_model.predict(original)
        if bundle["config"].get("fallback") == "forest_huber":
            huber = bundle["huber_models"][number]
            hp = np.asarray(huber["model"].predict(huber["builder"].transform(original, categorical_codes=True)), dtype=float)
            alpha = bundle["config"]["fallback_weight"]
            base = (1-alpha)*base + alpha*hp
        historical = _predict_history(history_model, augmented)
        predictions.append(np.where(known, historical, base))
    result = predictions[0] if len(predictions) == 1 else np.mean(predictions, axis=0)
    if result.shape != (len(original),) or not np.isfinite(result).all():
        raise ValueError("Маршрутизатор вернул некорректные предсказания")
    return result


def _split_indices(indices, size, name):
    result = np.asarray(indices)
    if (result.ndim != 1 or result.dtype.kind not in "iu"
            or (len(result) and (result.min() < 0 or result.max() >= size))
            or len(np.unique(result)) != len(result)):
        raise ValueError("Некорректные индексы " + name)
    return result.astype(np.int64, copy=False)


def train_router(task, config, profile, train, test, y, tr, va, threads,
                 refit=False, fixed_iterations=None, time_limit=None):
    """Тот же протокол bundle/predictions/details, что у ``train_model``."""
    started = time.monotonic()
    config.validate(task)
    tr, va = _split_indices(tr, len(train), "tr"), _split_indices(va, len(train), "va")
    if not len(tr) or np.intersect1d(tr, va).size:
        raise ValueError("Обучающая часть пуста либо пересекается с проверкой")
    target = np.asarray(y, dtype=float)
    if target.ndim != 1 or len(target) != len(train) or not np.isfinite(target).all():
        raise ValueError("Таргет должен быть конечным одномерным массивом")
    if fixed_iterations is not None and (isinstance(fixed_iterations, bool)
            or not isinstance(fixed_iterations, int) or not 1 <= fixed_iterations <= 6000):
        raise ValueError("Неверное фиксированное число итераций истории")
    if refit and fixed_iterations is None:
        raise ValueError("Для confirm/refit нужно фиксированное число итераций истории")
    if time_limit is not None and (isinstance(time_limit, bool)
            or not isinstance(time_limit, (int, float)) or not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("Лимит обучения должен быть положительным конечным числом")
    deadline = started + time_limit if time_limit is not None else float("inf")

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("Не завершён исторический маршрутизатор в выделенное время")
        return seconds if math.isfinite(seconds) else None

    original_train, original_test = train[task.feature_columns], test[task.feature_columns]
    schema = _task_schema(config.history_schema, task)
    history = HistoricalFeatures(schema).fit(original_train.iloc[tr], target[tr])
    numeric = [c for c in task.feature_columns if c not in task.categorical_columns]
    policy = (None if config.history_policy == "native" else
              RouterNumericPolicy(config.history_policy, numeric).fit(original_train.iloc[tr]))
    xtr = _augment(original_train.iloc[tr], history, policy)
    xva = _augment(original_train.iloc[va], history, policy)
    xte = _augment(original_test, history, policy)
    # Позиционная сборка сохраняет порядок tr и работает с повторными индексами.
    augmented = pd.concat([xtr, xva], ignore_index=True)
    augmented_target = np.concatenate([target[tr], target[va]])
    local_tr = np.arange(len(tr), dtype=np.int64)
    local_va = np.arange(len(tr), len(tr) + len(va), dtype=np.int64)
    augmented_task = replace(task, feature_columns=list(xtr.columns))
    fixed_fit = bool(refit or fixed_iterations is not None or not len(va))
    base_fields = {f.name: getattr(config, f.name) for f in fields(ExperimentSpec)}
    base_fields["model"] = "catboost"
    history_models, forest_models, huber_models, members = [], [], [], []
    validation_predictions, test_predictions = [], []
    selected_iterations = fixed_iterations
    for offset in range(config.seed_count):
        remaining()
        seed = config.seed + offset
        forest_spec = NovelSpec(seed=seed, forest_trees=256, forest_depth=24,
                                min_leaf=4, max_features=1.)
        forest = AlternativeModel(task, forest_spec, threads).fit(original_train.iloc[tr], target[tr])
        base = ExperimentSpec(**{**base_fields, "seed": seed})
        cb_bundle, cb_valid, cb_test, details = train_model(
            augmented_task, base, profile, augmented, xte, augmented_target,
            local_tr, local_va, threads, fixed_fit or offset > 0,
            selected_iterations, remaining())
        if offset == 0:
            selected_iterations = int(details["best_iterations"])
        if (fixed_fit or offset > 0) and (details.get("training_stopped_for_budget")
                or int(details["best_iterations"]) != selected_iterations):
            raise TimeoutError("Не завершён фиксированный CatBoost-компонент маршрутизатора")
        history_models.append(cb_bundle)
        forest_models.append(forest)
        forest_valid = forest.predict(original_train.iloc[va])
        forest_test = forest.predict(original_test)
        if config.fallback == "forest_huber":
            from proposal_models import ProposalSpec, train_proposal
            huber_config = ProposalSpec(variant="huber", seed=seed,
                base_config=ExperimentSpec(model="lightgbm", iterations=config.fallback_iterations, seed=seed).to_dict())
            hb, huber_valid, huber_test, hd = train_proposal(task, huber_config, profile, train, test, target,
                tr, va, threads, refit=True, fixed_iterations=config.fallback_iterations, time_limit=remaining())
            if hd.get("training_stopped_for_budget") or hd["best_iterations"] != config.fallback_iterations:
                raise TimeoutError("Не завершён фиксированный Huber-компонент")
            huber_models.append(hb)
            forest_valid = (1-config.fallback_weight)*forest_valid + config.fallback_weight*huber_valid
            forest_test = (1-config.fallback_weight)*forest_test + config.fallback_weight*huber_test
        validation_predictions.append(np.where(xva["__history_known"].to_numpy(dtype=bool),
                                                cb_valid, forest_valid))
        test_predictions.append(np.where(xte["__history_known"].to_numpy(dtype=bool),
                                         cb_test, forest_test))
        members.append({"seed": seed, "best_iterations": int(details["best_iterations"]),
                        "n_train": len(tr), "evaluation_used_for_fit": bool(len(va) and not fixed_fit and offset == 0),
                        "training_stopped_for_budget": bool(details.get("training_stopped_for_budget")),
                        "forest_config": forest_spec.to_dict()})
        remaining()
    trained = time.monotonic()
    bundle = {"kind": "history_router", "version": 1, "task": task.to_dict(),
              "config": config.to_dict(), "history": history, "policy": policy,
              "history_models": history_models, "forest_models": forest_models,
              "huber_models": huber_models}
    valid = (validation_predictions[0] if len(validation_predictions) == 1
             else np.mean(validation_predictions, axis=0))
    predicted = test_predictions[0] if len(test_predictions) == 1 else np.mean(test_predictions, axis=0)
    if not np.isfinite(valid).all() or not np.isfinite(predicted).all():
        raise ValueError("Маршрутизатор вернул неконечные предсказания")
    diagnostic = _feature_diagnostics(history, original_train.iloc[tr])
    history.transform(original_train.iloc[va])
    diagnostic["validation"] = dict(history.last_diagnostics_)
    history.transform(original_test)
    diagnostic["test"] = dict(history.last_diagnostics_)
    remaining()
    details = {"best_iterations": int(selected_iterations), "n_train": len(tr),
               "feature_count": len(xtr.columns), "training_seconds": trained - started,
               "prediction_seconds": time.monotonic() - trained,
               "component_fits": (3 if config.fallback == "forest_huber" else 2) * config.seed_count,
               "component_iterations": {"history": [m["best_iterations"] for m in members],
                                        "forest": [256] * config.seed_count,
                                        **({"huber": [config.fallback_iterations] * len(huber_models)} if huber_models else {})},
               "training_stopped_for_budget": any(m["training_stopped_for_budget"] for m in members),
               "evaluation_used_for_fit": members[0]["evaluation_used_for_fit"],
               "diagnostics": {"history": diagnostic, "history_policy": config.history_policy,
                               "numeric_policy_fit_rows": len(tr) if policy is not None else 0,
                               "members": members}}
    return bundle, valid, predicted, details
