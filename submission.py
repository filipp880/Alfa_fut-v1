"""Проверка и атомарная запись ответа без изменения идентификаторов."""
from __future__ import annotations
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from contracts import TaskSpec
from metrics import normalized_probabilities


def make_submission(task: TaskSpec, test: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    if len(prediction) != len(test):
        raise ValueError("Число предсказаний не совпадает с тестом")
    result = test[task.id_columns].copy().reset_index(drop=True)
    columns = [c for c in task.output_columns if c not in task.id_columns]
    if task.classification:
        probability = normalized_probabilities(prediction, len(task.classes))
        if task.prediction_kind == "probabilities":
            mapping = {str(c): i for i, c in enumerate(task.classes)}
            indices = task.probability_indices or [mapping[c] for c in task.probability_columns]
            for col, index in zip(task.probability_columns, indices):
                result[col] = probability[:, index]
        elif task.prediction_kind == "probability":
            result[columns[0]] = probability[:, task.positive_index]
        else:
            classes = np.asarray(task.classes)
            result[columns[0]] = classes[probability.argmax(axis=1)]
    else:
        pred = np.asarray(prediction, dtype=float).reshape(-1)
        if not np.isfinite(pred).all():
            raise ValueError("NaN или бесконечность в ответе")
        if task.metric == "rmsle":
            pred = np.maximum(pred, 0)
        result[columns[0]] = pred
    result = result[task.output_columns]
    if result.isna().any().any() or len(result) != len(test):
        raise ValueError("Пропуски или неверный размер ответа")
    if task.id_columns and result.duplicated(task.id_columns).any():
        raise ValueError("Дубли идентификаторов ответа")
    return result


def write_submission(task: TaskSpec, test: pd.DataFrame, prediction: np.ndarray, path: Path) -> None:
    frame = make_submission(task, test, prediction)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", suffix=".tmp",
                                         prefix=".submission-", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        # Проверяем уже сериализованный файл: схема и точность идентификаторов.
        reopened = pd.read_csv(temporary, dtype={c: str for c in task.id_columns},
                               keep_default_na=False, encoding="utf-8")
        if reopened.columns.tolist() != task.output_columns or len(reopened) != len(test):
            raise ValueError("Ошибка сериализации CSV")
        for col in task.id_columns:
            if reopened[col].tolist() != test[col].astype(str).tolist():
                raise ValueError("Изменён порядок или значение идентификаторов")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def fallback_prediction(task: TaskSpec, y: np.ndarray, n: int) -> np.ndarray:
    if task.classification:
        counts = np.bincount(y, minlength=len(task.classes)).astype(float)
        probability = counts / counts.sum()
        return np.tile(probability, (n, 1))
    value = float(np.mean(y))
    if task.metric == "mae":
        value = float(np.median(y))
    elif task.metric == "rmsle":
        value = float(np.expm1(np.mean(np.log1p(np.maximum(y, 0)))))
    return np.full(n, value)
