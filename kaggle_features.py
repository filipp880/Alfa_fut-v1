"""Ограниченные преобразования X: модельная импьютация и групповой PCA.

Все оценки, категории, масштабы и модели обучаются только в fit. Целевая
переменная не принимается API; вызывающий код передаёт target/id в forbidden.
Внешние схемы содержат только имена доступных признаков и числовые лимиты.
"""
from __future__ import annotations

from copy import deepcopy
import re
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler


def _numeric(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)


def _category(series):
    return series.astype(object).where(series.notna(), "__MISSING__").astype(str)


class _PredictorEncoder:
    """Маленькая плотная матрица без one-hot разрастания и target encoding."""

    def fit(self, frame):
        self.columns = list(frame.columns)
        self.numeric = {}
        self.categories = {}
        for column in self.columns:
            if pd.api.types.is_numeric_dtype(frame[column]):
                values = _numeric(frame[column])
                self.numeric[column] = float(values.median()) if values.notna().any() else 0.
            else:
                values = _category(frame[column])
                self.categories[column] = {v: i for i, v in enumerate(sorted(values.unique()))}
        return self

    def transform(self, frame):
        matrix = np.empty((len(frame), len(self.columns)), dtype=np.float32)
        for i, column in enumerate(self.columns):
            if column in self.numeric:
                values = _numeric(frame[column]).fillna(self.numeric[column]).to_numpy()
            else:
                values = _category(frame[column]).map(
                    self.categories[column]).fillna(-1).to_numpy()
            matrix[:, i] = np.clip(values, -1e30, 1e30)
        return matrix


class FoldFeatureAugmenter:
    """fit_transform(X) / transform(X) сохраняют порядок строк и исходные X.

    impute_columns: до четырёх числовых столбцов, каждый восстанавливается
    независимым ExtraTrees по другим X; исходный индикатор пропуска сохраняется.
    pca_groups: до четырёх групп, от 2 до 64 столбцов; добавляет компоненты,
    сохраняя исходные признаки. PCA использует исходные X с медианой train,
    поэтому его эффект отделим от выбранной модельной импьютации.
    """

    def __init__(self, schema=None, forbidden_columns=(), seed=42, threads=1):
        self.schema = deepcopy(schema or {})
        self.forbidden_columns = tuple(forbidden_columns)
        self.seed = seed
        self.threads = threads
        self.fitted_ = False

    def validate_columns(self, columns, categorical_columns=()):
        """Проверить DSL до создания ML-процесса без чтения наблюдений."""
        categorical = set(categorical_columns)
        empty = pd.DataFrame({c: pd.Series(dtype=object if c in categorical else float) for c in columns})
        self._validate(empty)
        return self

    def _validate(self, frame):
        allowed = {"pca_groups", "impute_columns", "imputer_estimators",
                   "imputer_min_samples_leaf", "imputer_max_rows", "imputer_max_depth"}
        if not isinstance(self.schema, dict) or set(self.schema) - allowed:
            raise ValueError("Недопустимая схема преобразований")
        if not frame.columns.is_unique or not all(isinstance(c, str) for c in frame.columns):
            raise ValueError("Требуются уникальные строковые имена признаков")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**31:
            raise ValueError("Недопустимый seed")
        if isinstance(self.threads, bool) or not isinstance(self.threads, int) or not 1 <= self.threads <= 8:
            raise ValueError("Допустимо от 1 до 8 потоков")
        features = set(frame) - set(self.forbidden_columns)
        for key, default, low, high in (("imputer_estimators", 80, 8, 200),
                ("imputer_min_samples_leaf", 5, 1, 100),
                ("imputer_max_rows", 30000, 50, 100000),
                ("imputer_max_depth", 16, 2, 24)):
            value = self.schema.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError("Недопустимый лимит " + key)
        impute = self.schema.get("impute_columns", [])
        if (not isinstance(impute, list) or len(impute) > 4
                or any(not isinstance(c, str) for c in impute) or len(set(impute)) != len(impute)):
            raise ValueError("Допустимо до четырёх уникальных столбцов импьютации")
        groups = self.schema.get("pca_groups", [])
        if not isinstance(groups, list) or len(groups) > 4:
            raise ValueError("Допустимо до четырёх групп PCA")
        names = []
        selected = list(impute)
        for group in groups:
            if not isinstance(group, dict) or set(group) != {"name", "columns", "components"}:
                raise ValueError("Недопустимая схема группы PCA")
            name, columns, components = group["name"], group["columns"], group["components"]
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", name):
                raise ValueError("Недопустимое имя группы PCA")
            if (not isinstance(columns, list) or not 2 <= len(columns) <= 64
                    or any(not isinstance(c, str) for c in columns) or len(set(columns)) != len(columns)):
                raise ValueError("Группа PCA требует 2..64 уникальных столбца")
            if isinstance(components, bool) or not isinstance(components, int) or not 1 <= components <= min(8, len(columns)):
                raise ValueError("Недопустимое число компонент PCA")
            names.append(name)
            selected.extend(columns)
        if len(set(names)) != len(names):
            raise ValueError("Имена групп PCA повторяются")
        if any(not isinstance(c, str) or c not in features for c in selected):
            raise ValueError("Схема содержит неизвестный или запрещённый столбец")
        if any(not pd.api.types.is_numeric_dtype(frame[c]) for c in selected):
            raise ValueError("Импьютация и PCA требуют числовые столбцы")
        generated = [f"{c}__imputed_missing" for c in impute]
        generated.extend(f"__pca_{g['name']}_{i+1}" for g in groups for i in range(g["components"]))
        if set(generated) & set(frame):
            raise ValueError("Создаваемый признак уже существует")
        return impute, groups

    def fit(self, frame):
        started = time.monotonic()
        impute, groups = self._validate(frame)
        self.columns_ = [c for c in frame if c not in self.forbidden_columns]
        if not len(frame):
            raise ValueError("Пустая обучающая часть")
        if impute and (len(self.columns_) > 256 or len(frame) * len(self.columns_) > 20000000):
            raise ValueError("Матрица модельной импьютации превышает лимит")
        self.imputers_ = {}
        self.pca_ = []
        self.diagnostics_ = {"fit_rows": len(frame), "input_columns": self.columns_,
                             "imputation": {}, "pca": {}}
        rng = np.random.default_rng(self.seed)
        for column in impute:
            target = _numeric(frame[column])
            observed = np.flatnonzero(target.notna().to_numpy())
            median = float(target.median()) if len(observed) else 0.
            predictors = [c for c in self.columns_ if c != column]
            info = {"observed_rows": len(observed), "missing_rate": float(target.isna().mean()),
                    "predictors": predictors, "fallback": median}
            encoder = estimator = None
            if len(observed) >= 30 and predictors and target.iloc[observed].nunique() > 1:
                maximum = self.schema.get("imputer_max_rows", 30000)
                if len(observed) > maximum:
                    observed = np.sort(rng.choice(observed, maximum, replace=False))
                encoder = _PredictorEncoder().fit(frame.iloc[observed][predictors])
                estimator = ExtraTreesRegressor(
                    n_estimators=self.schema.get("imputer_estimators", 80),
                    min_samples_leaf=self.schema.get("imputer_min_samples_leaf", 5),
                    max_depth=self.schema.get("imputer_max_depth", 16),
                    max_features=.8, random_state=self.seed, n_jobs=self.threads)
                estimator.fit(encoder.transform(frame.iloc[observed]), target.iloc[observed])
                info.update(status="model", model_rows=len(observed))
            else:
                info.update(status="median", model_rows=0)
            self.imputers_[column] = (encoder, estimator, median)
            self.diagnostics_["imputation"][column] = info
        for group in groups:
            columns = group["columns"]
            values = frame[columns].apply(_numeric)
            medians = values.median().fillna(0.)
            scaler = StandardScaler()
            scaled = scaler.fit_transform(values.fillna(medians))
            components = min(group["components"], len(frame))
            pca = PCA(n_components=components, svd_solver="full")
            pca.fit(scaled)
            self.pca_.append((group["name"], columns, medians, scaler, pca))
            self.diagnostics_["pca"][group["name"]] = {
                "columns": columns, "components": components,
                "explained_variance_ratio": np.nan_to_num(pca.explained_variance_ratio_).tolist()}
        self.diagnostics_["fit_seconds"] = time.monotonic() - started
        self.fitted_ = True
        return self

    def transform(self, frame):
        if not self.fitted_:
            raise ValueError("Преобразователь не обучен")
        if any(c not in frame for c in self.columns_):
            raise ValueError("Отсутствует обучающий признак")
        result = frame[self.columns_].copy()
        for column, (encoder, estimator, median) in self.imputers_.items():
            values = _numeric(frame[column])
            missing = values.isna()
            result[column + "__imputed_missing"] = missing.astype(float)
            if missing.any():
                values.loc[missing] = estimator.predict(encoder.transform(frame.loc[missing])) if estimator is not None else median
            result[column] = values
        for name, columns, medians, scaler, pca in self.pca_:
            values = frame[columns].apply(_numeric).fillna(medians)
            if len(frame):
                transformed = pca.transform(scaler.transform(values))
            else:
                transformed = np.empty((0, pca.n_components_))
            for i in range(pca.n_components_):
                result[f"__pca_{name}_{i+1}"] = transformed[:, i]
        return result

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)


def regression_feature_schema(dictionary, columns):
    """Предложение схемы по описаниям словаря; не читает данные или target.

    dictionary — DataFrame со столбцами feature, role, dtype, description_ru.
    Неизвестные метаданные дают пустую схему. Решение об испытании этой схемы
    принимает исследовательский контроллер, а не этот вспомогательный метод.
    """
    if not isinstance(dictionary, pd.DataFrame) or not {"feature", "role", "dtype", "description_ru"} <= set(dictionary):
        return {"pca_groups": [], "impute_columns": []}
    allowed = set(columns)
    warm, cold, impute = [], [], []
    for row in dictionary.to_dict("records"):
        column = row["feature"]
        if column not in allowed or row["role"] != "feature" or row["dtype"] != "numeric":
            continue
        description = str(row["description_ru"]).lower()
        if ("максимальная температура" in description or "температурой выше" in description
                or "сумма превышений" in description):
            warm.append(column)
        if ("минимальная температура" in description or "температурой ниже" in description
                or "сумма недостатка" in description):
            cold.append(column)
        if "рейтинг энергоэффективности" in description or "год постройки" in description:
            impute.append(column)
    groups = [{"name": name, "columns": group, "components": 2}
              for name, group in (("warm", warm), ("cold", cold)) if 2 <= len(group) <= 64]
    return {"pca_groups": groups, "impute_columns": impute[:4]}
