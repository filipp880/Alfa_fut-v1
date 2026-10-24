"""Определение контракта из условия, словаря и структуры CSV."""
from __future__ import annotations

import csv
from pathlib import Path
import re
import numpy as np
import pandas as pd

from contracts import CLASSIFICATION_METRICS, TaskSpec, native


METRIC_PATTERNS = [
    ("rmsle", r"\brmsle\b|root mean squared log"),
    ("rmse", r"\brmse\b|root mean squared error|корень.*среднеквадратич"),
    ("macro_f1", r"macro[ _-]*f[ _-]*1|f[ _-]*1[ _-]*macro|average\s*=\s*[\"']macro"),
    ("weighted_f1", r"weighted[ _-]*f[ _-]*1|f[ _-]*1[ _-]*weighted"),
    ("balanced_accuracy", r"balanced[ _-]*accuracy|сбалансированн\w* точност"),
    ("log_loss", r"log[ _-]*loss|logarithmic loss|cross[ _-]*entropy"),
    ("average_precision", r"average[ _-]*precision|pr[ _-]*auc|auc[ _-]*pr"),
    ("roc_auc", r"roc[ _-]*auc|auc[ _-]*roc|\bauroc\b"),
    ("mape", r"\bmape\b|mean absolute percentage"),
    ("mae", r"\bmae\b|mean absolute error"),
    ("mse", r"\bmse\b|mean squared error"),
    ("r2", r"\br[ _-]?(?:2|²)\b|r2_score|коэффициент детерминации"),
    ("f1", r"\bf[ _-]*1(?:_score)?\b"),
    ("accuracy", r"\baccuracy\b"),
]


def find_file(root: Path, name: str, text: str, required: bool = True) -> Path | None:
    references = re.findall(r"`([^`\n]+\.csv)`", text, re.I)
    for reference in references:
        if Path(reference.replace("\\", "/")).name.lower() != name.lower():
            continue
        candidate = (root / reference).resolve()
        if candidate.is_relative_to(root) and candidate.is_file():
            return candidate
    candidate = root / name
    if candidate.is_file() and candidate.resolve().is_relative_to(root):
        return candidate.resolve()
    candidates = [p for p in root.rglob(name)
                  if ".alfa_runs" not in p.parts and p.resolve().is_relative_to(root)]
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not required and not candidates:
        return None
    raise ValueError(f"Не найден однозначный файл {name}: {len(candidates)} кандидатов")


def read_csv(path: str | Path, string_columns: list[str] | None = None) -> pd.DataFrame:
    with open(path, encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(header) != len(set(header)):
        raise ValueError(f"Пустой или неоднозначный заголовок CSV: {Path(path).name}")
    dtypes = {c: str for c in (string_columns or []) if c in header}
    return pd.read_csv(path, encoding="utf-8-sig", dtype=dtypes,
                       keep_default_na=False, na_values=[""])


def detect_metric(text: str) -> str | None:
    lines = text.lower().splitlines()
    # Основная метрика имеет приоритет над справочными и упоминаниями в описании.
    primary = [line for line in lines
               if re.search(r"основн\w* метрик|метрика оцен|primary metric|evaluation metric|main metric", line)]
    for scope in ["\n".join(primary), text.lower()]:
        for metric, pattern in METRIC_PATTERNS:
            if re.search(pattern, scope, re.I):
                return metric
    return None


def load_task(task_path: str | Path) -> tuple[TaskSpec, pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    path = Path(task_path).resolve()
    text = path.read_text(encoding="utf-8-sig")
    root = path.parent
    train_path = find_file(root, "train.csv", text)
    test_path = find_file(root, "test.csv", text)
    sample_path = find_file(root, "sample_submission.csv", text, False)
    dd_path = find_file(root, "data_dictionary.csv", text, False)
    dictionary = read_csv(dd_path).fillna("") if dd_path else pd.DataFrame()
    ids, categorical, targets, excluded = [], [], [], []
    if not dictionary.empty and {"feature", "role"} <= set(dictionary.columns):
        for row in dictionary.to_dict("records"):
            name, role = str(row["feature"]), str(row["role"]).lower().strip()
            if role in {"target", "label", "таргет", "цель"}:
                targets.append(name)
            if role in {"id", "identifier", "идентификатор"}:
                ids.append(name)
            if role in {"ignore", "excluded", "leakage", "drop", "исключить", "утечка"}:
                excluded.append(name)
            if str(row.get("dtype", "")).lower() in {"category", "categorical", "категория", "string"}:
                categorical.append(name)
    # Идентификаторы читаются строками: ведущие нули и длинные числа сохраняются.
    header = pd.read_csv(test_path, encoding="utf-8-sig", nrows=0).columns.tolist()
    ids = [c for c in ids if c in header]
    if not ids and sample_path:
        template_columns = pd.read_csv(sample_path, encoding="utf-8-sig", nrows=0).columns.tolist()
        shared = [c for c in template_columns if c in header]
        if shared:
            template_values = pd.read_csv(sample_path, usecols=shared, dtype=str, keep_default_na=False)
            test_values = pd.read_csv(test_path, usecols=shared, dtype=str, keep_default_na=False)
            ids = [c for c in shared if template_values[c].equals(test_values[c]) and test_values[c].is_unique]
    if not ids and not sample_path:
        ids = [c for c in header if c.lower() in {"id", "row_id", "sample_id"}]
    train = read_csv(train_path, ids + categorical)
    test = read_csv(test_path, ids + categorical)
    sample = read_csv(sample_path, ids) if sample_path else None
    targets = [c for c in targets if c in train]
    differences = [c for c in train if c not in test and c not in ids]
    if not targets:
        targets = differences
    if len(targets) != 1:
        raise ValueError(f"Ожидается один таргет; найдены {targets}")
    target = targets[0]
    if target in test or train[target].isna().any():
        raise ValueError("Таргет присутствует в тесте или содержит пропуски")
    if not len(train) or not len(test):
        raise ValueError("Пустая обучающая или тестовая выборка")
    ids = [c for c in ids if c in test and c != target]
    for col in ids:
        if test[col].isna().any():
            raise ValueError(f"Пропущен идентификатор {col}")
    if ids and test.duplicated(ids).any():
        raise ValueError("Повторяющиеся идентификаторы теста")
    features = [c for c in test if c in train and c not in ids and c not in excluded and c != target]
    if not features:
        raise ValueError("Нет общих признаков")
    categorical = [c for c in categorical if c in features]
    categorical += [c for c in features if c not in categorical and not pd.api.types.is_numeric_dtype(train[c])]
    metric = detect_metric(text)
    warnings = []
    if metric in CLASSIFICATION_METRICS:
        problem = "classification"
    elif metric:
        problem = "regression"
    elif re.search(r"классификац|classification", text, re.I):
        problem = "classification"
    elif re.search(r"регресси|regression", text, re.I):
        problem = "regression"
    else:
        problem = "classification" if not pd.api.types.is_numeric_dtype(train[target]) else "regression"
        warnings.append("Тип задачи выведен из данных; требуется проверка LLM")
    if metric is None:
        metric = "unknown"
        warnings.append("Не удалось определить поддерживаемую метрику")
    classes = sorted(native(train[target].unique().tolist()), key=lambda x: str(x)) if problem == "classification" else []
    if len(classes) > 500:
        raise ValueError("Более 500 классов; проверьте постановку")
    output_columns = sample.columns.tolist() if sample is not None else ids + [target]
    # Идентификатор может быть нужен только для исключения из признаков,
    # а шаблон ответа при этом может содержать один столбец предсказаний.
    ids = [c for c in ids if c in output_columns]
    prediction_columns = [c for c in output_columns if c not in ids]
    kind, probability_columns, probability_indices = ("label" if problem == "classification" else "value"), [], []
    if problem == "classification":
        exact_labels = {str(c): i for i, c in enumerate(classes)}
        label_to_indices = {}
        for i, value in enumerate(classes):
            label_to_indices.setdefault(str(value).lower(), []).append(i)
        mapped = []
        for col in prediction_columns:
            name = col
            if name not in exact_labels:
                name = re.sub(r"^(?:probability|prob|proba|class|p)[ _-]+", "", name, flags=re.I)
            matches = label_to_indices.get(name.lower(), [])
            mapped.append(exact_labels[name] if name in exact_labels else matches[0] if len(matches) == 1 else None)
        if len(prediction_columns) == len(classes) and None not in mapped and len(set(mapped)) == len(classes):
            kind, probability_columns, probability_indices = "probabilities", prediction_columns, mapped
        elif len(prediction_columns) == 1 and (metric in {"roc_auc", "average_precision", "log_loss"}
                or re.search(r"предсказ\w* вероятност|probability predictions|predict probabilities", text, re.I)):
            if len(classes) != 2:
                raise ValueError("Для многоклассовых вероятностей нужен шаблон с колонками классов")
            kind = "probability"
    if kind not in {"probabilities"} and len(prediction_columns) != 1:
        raise ValueError("Неподдерживаемая схема ответа")
    if sample is not None and len(sample) != len(test):
        raise ValueError("Размер шаблона не совпадает с тестом")
    spec = TaskSpec(str(path), str(train_path), str(test_path), str(sample_path) if sample_path else None,
                    str(dd_path) if dd_path else None, target, problem, metric, ids, features,
                    categorical, output_columns, kind, classes, probability_columns, warnings=warnings)
    spec.probability_indices = probability_indices
    if len(classes) == 2:
        match = re.search(r"(?:positive\s+(?:class|label)|положительн\w*\s+(?:класс\w*|метк\w*)|"
                          r"вероятност[ьи]\s+класса)\s*(?:[:=—–-]|is|это)?\s*[`\"']?([^`\"'\s,.;]+)", text, re.I)
        if match:
            label = match.group(1).lower()
            matches = [i for i, value in enumerate(classes) if str(value).lower() == label]
            if matches:
                spec.positive_index = matches[0]
    return spec, train, test, sample, dictionary
