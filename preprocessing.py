"""Преобразования обучаются только на train-части конкретного эксперимента."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from contracts import ExperimentSpec, TaskSpec


class FeatureBuilder:
    def __init__(self, task: TaskSpec, config: ExperimentSpec, profile: dict):
        self.task = task
        self.config = config
        self.time_column = profile.get("time_column")
        self.columns = [c for c in task.feature_columns
                        if not (config.drop_time and c == self.time_column)]
        self.categorical = [c for c in task.categorical_columns if c in self.columns]
        self.date_columns = []
        self.frequencies = {}
        self.levels = {}
        self.indicators = []
        self.output_columns = []

    def fit(self, frame: pd.DataFrame):
        for col in list(self.categorical):
            sample = frame[col].dropna().astype(str).head(100)
            if len(sample) and sample.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").mean() >= .95:
                self.date_columns.append(col)
                self.categorical.remove(col)
        if self.config.missing_indicators:
            self.indicators = [c for c in self.columns if frame[c].isna().any()]
        for col in self.categorical:
            values = frame[col].fillna("__MISSING__").astype(str)
            self.levels[col] = sorted(values.unique().tolist())
            if self.config.frequency_encoding:
                self.frequencies[col] = values.value_counts(normalize=True).to_dict()
        for transform in self.config.transforms:
            if transform["op"] != "missing":
                for col in transform["columns"]:
                    if col in self.categorical or col in self.date_columns:
                        raise ValueError("Числовая операция над категориальным признаком")
            if transform["op"] == "log1p":
                values = pd.to_numeric(frame[transform["columns"][0]], errors="coerce")
                if (values.dropna() < 0).any():
                    raise ValueError("log1p разрешён только для неотрицательного признака")
        self.output_columns = self.transform(frame.head(2)).columns.tolist()
        return self

    def transform(self, frame: pd.DataFrame, categorical_codes: bool = False) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        for col in self.columns:
            if col in self.date_columns:
                date = pd.to_datetime(frame[col], errors="coerce", utc=True)
                result[col+"__year"] = date.dt.year.astype(float)
                result[col+"__month"] = date.dt.month.astype(float)
                result[col+"__day"] = date.dt.day.astype(float)
                result[col+"__weekday"] = date.dt.dayofweek.astype(float)
            elif col in self.categorical:
                values = frame[col].fillna("__MISSING__").astype(str)
                if categorical_codes:
                    result[col] = pd.Categorical(values, categories=self.levels[col])
                else:
                    result[col] = values
                if col in self.frequencies:
                    result[col+"__frequency"] = values.map(self.frequencies[col]).fillna(0).astype(float)
            else:
                result[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
        for col in self.indicators:
            result[col+"__missing"] = frame[col].isna().astype(float)
        for index, transform in enumerate(self.config.transforms):
            op, columns = transform["op"], transform["columns"]
            name = f"__derived_{index}_{op}"
            if op == "missing":
                result[name] = frame[columns[0]].isna().astype(float)
                continue
            a = pd.to_numeric(frame[columns[0]], errors="coerce").to_numpy(dtype=float)
            if op == "log1p":
                result[name] = np.log1p(np.where(a >= 0, a, np.nan))
                continue
            b = pd.to_numeric(frame[columns[1]], errors="coerce").to_numpy(dtype=float)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                if op == "ratio":
                    value = np.divide(a, b, out=np.full_like(a, np.nan), where=np.abs(b) > 1e-8)
                elif op == "difference":
                    value = a-b
                else:
                    value = a+b
            result[name] = np.where(np.isfinite(value), value, np.nan)
        if result.empty and len(frame):
            result["__constant"] = 1.0
        return result
