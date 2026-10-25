"""История повторных сущностей: только метки строго более ранних периодов.

Схема задаётся по словарю задачи, а не по имени датасета. Пропуск хотя бы
одной части ключа запрещает сопоставление. Совпадение составного ключа само
по себе не доказывает, что строки относятся к одной физической сущности.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numbers

import numpy as np
import pandas as pd


HISTORY_COLUMNS = ("__history_mean", "__history_last", "__history_count",
                   "__history_age", "__history_known")


@dataclass(frozen=True)
class HistorySchema:
    time_column: str
    entity_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    forbidden_columns: tuple[str, ...] = ()
    min_count: int = 1

    def validate(self) -> "HistorySchema":
        if not isinstance(self.time_column, str) or not self.time_column:
            raise ValueError("Нужен столбец упорядоченного периода")
        for values in (self.entity_columns, self.feature_columns, self.forbidden_columns):
            if not isinstance(values, (list, tuple)) or any(
                    not isinstance(c, str) or not c for c in values):
                raise ValueError("Имена столбцов должны быть непустыми строками")
            if len(values) != len(set(values)):
                raise ValueError("Повторяющиеся имена столбцов запрещены")
        if not 1 <= len(self.entity_columns) <= 8:
            raise ValueError("Допустимо от 1 до 8 частей ключа")
        used = {self.time_column, *self.entity_columns}
        if self.time_column in self.entity_columns:
            raise ValueError("Период не должен входить в идентификатор сущности")
        if not used <= set(self.feature_columns) or used & set(self.forbidden_columns):
            raise ValueError("Схема использует неизвестный, целевой или ID-столбец")
        if isinstance(self.min_count, bool) or not isinstance(self.min_count, int) or not 1 <= self.min_count <= 100:
            raise ValueError("min_count должен быть целым от 1 до 100")
        return self

    @classmethod
    def from_dict(cls, data: dict, feature_columns, forbidden_columns=()) -> "HistorySchema":
        if not isinstance(data, dict) or set(data) - {"time_column", "entity_columns", "min_count"}:
            raise ValueError("Недопустимая схема исторических признаков")
        return cls(time_column=data.get("time_column"),
                   entity_columns=data.get("entity_columns", ()),
                   feature_columns=tuple(feature_columns),
                   forbidden_columns=tuple(forbidden_columns),
                   min_count=data.get("min_count", 1)).validate()


def suggest_history_schema(dictionary, feature_columns, time_column, forbidden_columns=()):
    """Предложить ключ по метаданным повторно измеряемых объектов.

    Имена колонок не фиксируются. Это проверяемая гипотеза о группе, а не
    обещание идентификации физического объекта. Требуется числовой период.
    """
    import re
    if not time_column or not {"feature", "role", "dtype", "description_ru"} <= set(dictionary):
        return {}
    allowed = set(feature_columns)-set(forbidden_columns)
    rows = dictionary.to_dict("records")
    if not any(r["feature"] == time_column and r["dtype"] == "numeric" for r in rows):
        return {}
    patterns = [r"штат|регион|\bstate\b|\bregion\b", r"класс здания|building class",
                r"назначение здания|тип объекта|facility type|building use",
                r"площадь.*здан|floor area|building area", r"год постройки|year.*built|construction year"]
    chosen = []
    for pattern in patterns:
        matches = [r["feature"] for r in rows if r["feature"] in allowed and r["feature"] != time_column
                   and r["role"] == "feature" and re.search(pattern,str(r["description_ru"]),re.I)]
        if len(matches) != 1:
            return {}
        chosen.append(matches[0])
    schema = {"time_column":time_column,"entity_columns":list(dict.fromkeys(chosen)),
              "feature_columns":list(feature_columns),"forbidden_columns":list(forbidden_columns)}
    HistorySchema(**schema).validate()
    return schema


def _part(value):
    """Типизированный кортеж вместо неоднозначной конкатенации с разделителем."""
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return ("bool", bool(value))
    if isinstance(value, numbers.Real):
        if not math.isfinite(float(value)):
            return None
        return ("number", float(value))
    if isinstance(value, str):
        return ("string", value)
    raise ValueError("Для ключа поддерживаются только числовые и строковые значения")


class HistoricalFeatures:
    def __init__(self, schema: HistorySchema):
        self.schema = schema.validate()
        self.histories_ = None
        self.diagnostics_ = {}
        self.last_diagnostics_ = {}

    def _inputs(self, frame):
        required = [self.schema.time_column, *self.schema.entity_columns]
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
            raise ValueError("Ожидается DataFrame с уникальными столбцами")
        if any(c not in frame.columns for c in required):
            raise ValueError("В таблице отсутствуют столбцы схемы")
        if set(HISTORY_COLUMNS) & set(frame.columns):
            raise ValueError("Служебные столбцы истории уже существуют")
        time = pd.to_numeric(frame[self.schema.time_column], errors="coerce").to_numpy(dtype=float, copy=True)
        time[~np.isfinite(time)] = np.nan
        keys = []
        for row in frame[list(self.schema.entity_columns)].itertuples(index=False, name=None):
            parts = tuple(_part(value) for value in row)
            keys.append(None if any(value is None for value in parts) else parts)
        return time, keys

    def fit(self, frame: pd.DataFrame, y) -> "HistoricalFeatures":
        time, keys = self._inputs(frame)
        target = np.asarray(y, dtype=float)
        if target.ndim != 1 or len(target) != len(frame) or not np.isfinite(target).all():
            raise ValueError("Таргет истории должен быть конечным одномерным массивом")
        buckets = {}
        missing_key = missing_time = 0
        for key, period, value in zip(keys, time, target):
            missing_key += key is None
            missing_time += not np.isfinite(period)
            if key is not None and np.isfinite(period):
                buckets.setdefault(key, {}).setdefault(float(period), []).append(float(value))
        self.histories_ = {}
        ambiguous_groups = ambiguous_rows = conflicting_groups = 0
        for key, periods in buckets.items():
            ordered = sorted(periods)
            counts = np.array([len(periods[t]) for t in ordered], dtype=np.int64)
            sums = np.array([sum(periods[t]) for t in ordered], dtype=float)
            for values in periods.values():
                if len(values) > 1:
                    ambiguous_groups += 1
                    ambiguous_rows += len(values)
                    conflicting_groups += len(set(values)) > 1
            self.histories_[key] = (np.asarray(ordered), np.cumsum(counts),
                                    np.cumsum(sums), sums/counts)
        self.diagnostics_ = {"fit_rows": len(frame), "entity_keys": len(buckets),
                             "missing_key_rows": missing_key, "missing_time_rows": missing_time,
                             "multirow_key_period_groups": ambiguous_groups,
                             "multirow_key_period_rows": ambiguous_rows,
                             "different_target_key_period_groups": conflicting_groups,
                             "note": "Повторы ключа внутри периода — возможные коллизии либо повторные измерения; истинный ID неизвестен."}
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.histories_ is None:
            raise ValueError("Сначала нужно обучить историю")
        time, keys = self._inputs(frame)
        values = np.full((len(frame), len(HISTORY_COLUMNS)), np.nan, dtype=float)
        values[:, 2] = 0
        values[:, 4] = 0
        unknown_keys = no_past = 0
        for i, (key, period) in enumerate(zip(keys, time)):
            history = self.histories_.get(key) if key is not None else None
            if history is None:
                unknown_keys += 1
                continue
            if not np.isfinite(period):
                no_past += 1
                continue
            periods, counts, sums, means = history
            # side='left' excludes every label from this same period.
            position = np.searchsorted(periods, period, side="left")-1
            if position < 0 or counts[position] < self.schema.min_count:
                no_past += 1
                continue
            values[i] = [sums[position]/counts[position], means[position],
                         counts[position], period-periods[position], 1.]
        result = frame.copy()
        for i, name in enumerate(HISTORY_COLUMNS):
            result[name] = values[:, i]
        known = int(values[:, 4].sum())
        self.last_diagnostics_ = {"rows": len(frame), "known_history_rows": known,
                                 "coverage": known/len(frame) if len(frame) else 0.,
                                 "unknown_key_rows": unknown_keys, "no_earlier_history_rows": no_past}
        return result

    def fit_transform(self, frame: pd.DataFrame, y) -> pd.DataFrame:
        result = self.fit(frame, y).transform(frame)
        self.diagnostics_["training_history"] = dict(self.last_diagnostics_)
        return result
