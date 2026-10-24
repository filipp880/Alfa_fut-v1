"""Профиль данных: без изменения таблиц и без использования ответов теста."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

from contracts import TaskSpec, native


def encode_target(task: TaskSpec, train: pd.DataFrame) -> np.ndarray:
    if task.classification:
        mapping = {str(value): index for index, value in enumerate(task.classes)}
        return train[task.target].map(lambda value: mapping[str(value)]).to_numpy(dtype=np.int64)
    y = pd.to_numeric(train[task.target], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Неконечный таргет регрессии")
    return y


def feature_groups(task: TaskSpec, train: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(train[task.feature_columns], index=False).to_numpy()


def detect_time(task: TaskSpec, train: pd.DataFrame, test: pd.DataFrame,
                dictionary: pd.DataFrame) -> tuple[str | None, np.ndarray | None]:
    descriptions = {}
    if not dictionary.empty and "feature" in dictionary:
        descriptions = {str(row["feature"]): str(row.get("description_ru", ""))
                        for row in dictionary.to_dict("records")}
    candidates = []
    for col in task.feature_columns:
        semantic = col + " " + descriptions.get(col, "")
        if not re.search(r"year|date|time|period|год|дат[аы]|период", semantic, re.I):
            continue
        if re.search(r"built|birth|construction|построй|рожден|создан", semantic, re.I):
            continue
        a, b = train[col], test[col]
        if pd.api.types.is_numeric_dtype(a):
            # Коды периодов: небольшое число уровней, не длительность/время отклика.
            timestamp = bool(re.search(r"timestamp|datetime|(?:^|_)date(?:_|$)|event_time", col, re.I))
            levels = a.nunique()
            if levels < 3 or (levels > 100 and not timestamp) or a.isna().any() or b.isna().any():
                continue
            av, bv = a.to_numpy(dtype=float), pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
        else:
            values = a.dropna().astype(str).head(64)
            if values.empty or values.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").mean() < .9:
                continue
            aa, bb = pd.to_datetime(a, errors="coerce", utc=True), pd.to_datetime(b, errors="coerce", utc=True)
            if aa.isna().any() or bb.isna().any():
                continue
            av, bv = aa.astype("int64").to_numpy(), bb.astype("int64").to_numpy()
        if np.isfinite(bv).all() and np.min(bv) > np.max(av):
            candidates.append((col, av))
    return candidates[0] if candidates else (None, None)


def profile_data(task: TaskSpec, train: pd.DataFrame, test: pd.DataFrame,
                 dictionary: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray | None]:
    groups = feature_groups(task, train)
    time_column, time_values = detect_time(task, train, test, dictionary)
    cats = task.categorical_columns
    text_columns = [c for c in cats if train[c].astype(str).str.len().median() > 40
                    and train[c].nunique() > min(100, len(train) * .2)]
    result = {
        "train_rows": len(train), "test_rows": len(test),
        "feature_count": len(task.feature_columns), "categorical_columns": cats,
        "text_columns": text_columns, "time_column": time_column,
        "time_levels": int(len(np.unique(time_values))) if time_values is not None else 0,
        "duplicate_rows": int(len(train) - len(np.unique(groups))),
        "missing_train": {c: round(float(train[c].isna().mean()), 4)
                          for c in task.feature_columns if train[c].isna().any()},
        "missing_test": {c: round(float(test[c].isna().mean()), 4)
                         for c in task.feature_columns if test[c].isna().any()},
        "categorical_cardinality": {c: int(train[c].nunique()) for c in cats},
        "unseen_category_fraction": {
            c: round(float((~test[c].isin(set(train[c].dropna())) & test[c].notna()).mean()), 4) for c in cats},
        "train_memory_mb": round(float(train.memory_usage(deep=True).sum()/2**20), 2),
    }
    if task.classification:
        result["class_counts"] = native(train[task.target].value_counts().to_dict())
        result["minority_fraction"] = float(train[task.target].value_counts(normalize=True).min())
    else:
        y = encode_target(task, train)
        result["target_summary"] = {
            "min": float(y.min()), "max": float(y.max()), "mean": float(y.mean()),
            "median": float(np.median(y)), "nonnegative": bool(y.min() >= 0),
        }
    return result, groups, time_values
