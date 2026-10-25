"""Ограниченный CPU XGBoost; все уровни категорий известны только из fit-части.

Исследовательский кандидат, а не автоматическое включение в штатный портфель.
Документация API сверена с исходниками XGBoost v2.1.4.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from contracts import ExperimentSpec, TaskSpec, fingerprint
from metrics import evaluate, normalized_probabilities
from models import sample_weights
from preprocessing import FeatureBuilder


@dataclass
class XGBSpec:
    model: str = "xgboost"
    iterations: int = 1200
    depth: int = 6
    learning_rate: float = .06
    regularization: float = 5.
    min_child_weight: float = 5.
    class_weight_power: float = 0.
    missing_indicators: bool = False
    frequency_encoding: bool = False
    drop_time: bool = False
    seed: int = 42
    hypothesis: str = "Прямой CPU XGBoost с выбором итераций по целевой метрике"

    def validate(self, task: TaskSpec) -> "XGBSpec":
        if self.model != "xgboost":
            raise ValueError("Недопустимая модель XGBoost")
        bounds = {"iterations": (1, 6000), "depth": (2, 9),
                  "learning_rate": (.01, .3), "regularization": (.01, 100.),
                  "min_child_weight": (.01, 100.), "class_weight_power": (0., 1.),
                  "seed": (0, 2**31-1)}
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Неверный числовой параметр {name}")
            if name in {"iterations", "depth", "seed"} and not isinstance(value, int):
                raise ValueError(f"Ожидается целое {name}")
            if not low <= value <= high:
                raise ValueError(f"{name} вне допустимого диапазона")
        for name in ("missing_indicators", "frequency_encoding", "drop_time"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"Ожидается bool: {name}")
        if not task.classification and self.class_weight_power:
            raise ValueError("Веса классов неприменимы к регрессии")
        if not isinstance(self.hypothesis, str) or len(self.hypothesis) > 1200:
            raise ValueError("Недопустимое описание гипотезы")
        if not task.feature_columns or task.target in task.feature_columns or set(task.id_columns) & set(task.feature_columns):
            raise ValueError("Признаки XGBoost содержат target/id или пусты")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def key(self) -> str:
        data = self.to_dict()
        data.pop("hypothesis")
        return fingerprint(data)

    @classmethod
    def from_dict(cls, data: dict, task: TaskSpec) -> "XGBSpec":
        if not isinstance(data, dict):
            raise ValueError("Конфигурация XGBoost должна быть объектом")
        return cls(**data).validate(task)


def _make_builder(task: TaskSpec, config: XGBSpec, profile: dict, frame: pd.DataFrame) -> FeatureBuilder:
    base = ExperimentSpec(model="lightgbm", missing_indicators=config.missing_indicators,
                          frequency_encoding=config.frequency_encoding, drop_time=config.drop_time,
                          seed=config.seed)
    return FeatureBuilder(task, base, profile).fit(frame)


def class_metrics(task: TaskSpec, y: np.ndarray, prediction: np.ndarray) -> dict:
    """Macro F1 не скрывает поведение редких классов; метрики без весов."""
    if not task.classification:
        return {}
    labels = np.arange(len(task.classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        np.asarray(y, dtype=int), np.argmax(prediction, axis=1), labels=labels,
        zero_division=0)
    return {str(task.classes[i]): {"precision": float(precision[i]),
             "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i in labels}


def train_xgboost(task: TaskSpec, config: XGBSpec | dict, profile: dict,
                  train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray,
                  train_indices: np.ndarray, valid_indices: np.ndarray,
                  threads: int, refit: bool = False, fixed_iterations: int | None = None,
                  time_limit: float | None = None) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    """Интерфейс models.train_model; refit никогда не передаёт valid target в xgb.train.

    При поиске early stopping оптимизирует именно TaskSpec.metric, включая Macro F1.
    На подтверждении/refit используется заранее зафиксированное число итераций.
    Неизвестные категории становятся NaN с сохранением кодов категорий fit-части.
    """
    import xgboost as xgb

    started = time.monotonic()
    config = XGBSpec.from_dict(config, task) if isinstance(config, dict) else config.validate(task)
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 7:
        raise ValueError("XGBoost использует от 1 до 7 потоков")
    if fixed_iterations is not None and (isinstance(fixed_iterations, bool)
            or not isinstance(fixed_iterations, int) or not 1 <= fixed_iterations <= config.iterations):
        raise ValueError("Неверное зафиксированное число итераций")
    tr, va = np.asarray(train_indices), np.asarray(valid_indices)
    if any(a.ndim != 1 or a.dtype.kind not in "iu" for a in (tr, va)):
        raise ValueError("Индексы разбиения должны быть целочисленным вектором")
    if not len(tr) or np.intersect1d(tr, va).size or len(np.unique(tr)) != len(tr):
        raise ValueError("Некорректное разбиение XGBoost")
    if any((a < 0).any() or (a >= len(train)).any() for a in (tr, va)):
        raise ValueError("Индексы вне таблицы")
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("Неверный временной лимит")
    deadline = started + time_limit if time_limit is not None else float("inf")
    builder = _make_builder(task, config, profile, train.iloc[tr])
    x_train = builder.transform(train.iloc[tr], categorical_codes=True)
    x_valid = builder.transform(train.iloc[va], categorical_codes=True) if len(va) else None
    x_test = builder.transform(test, categorical_codes=True)
    # Train-only FeatureBuilder levels guarantee the same category/code mapping.
    dtrain = xgb.DMatrix(x_train, label=y[tr], enable_categorical=True, nthread=threads,
                        weight=sample_weights(y[tr], len(task.classes), config.class_weight_power)
                        if task.classification else None)
    evaluating = bool(len(va) and not refit)
    dvalid = (xgb.DMatrix(x_valid, label=y[va] if evaluating else None,
                         enable_categorical=True, nthread=threads) if len(va) else None)
    params = {"tree_method": "hist", "device": "cpu", "nthread": threads,
              "max_depth": config.depth, "eta": config.learning_rate,
              "lambda": config.regularization, "min_child_weight": config.min_child_weight,
              "subsample": .85, "colsample_bytree": .9, "max_bin": 256,
              "seed": config.seed, "verbosity": 0, "disable_default_eval_metric": 1,
              "validate_parameters": True}
    if task.classification:
        if len(task.classes) < 2:
            raise ValueError("Классификация требует не менее двух классов")
        params.update(objective="multi:softprob", num_class=len(task.classes))
    else:
        params["objective"] = "reg:absoluteerror" if task.metric == "mae" else "reg:squarederror"

    class Deadline(xgb.callback.TrainingCallback):
        stopped = False

        def after_iteration(self, model, epoch, evals_log):
            self.stopped = time.monotonic() >= deadline
            return self.stopped

    stop = Deadline()

    def custom_metric(prediction, matrix):
        values = normalized_probabilities(prediction, len(task.classes)) if task.classification else prediction
        if task.classification and task.metric == "macro_f1":
            return "macro_f1", float(f1_score(matrix.get_label(), values.argmax(axis=1),
                labels=np.arange(len(task.classes)), average="macro", zero_division=0))
        return task.metric, float(evaluate(task, matrix.get_label(), values)[task.metric])

    if time.monotonic() >= deadline:
        raise TimeoutError("Время XGBoost истекло на подготовке данных")
    iterations = fixed_iterations if fixed_iterations is not None else config.iterations
    model = xgb.train(params, dtrain, num_boost_round=iterations,
                      evals=[(dvalid, "search")] if evaluating else [],
                      custom_metric=custom_metric if evaluating else None,
                      early_stopping_rounds=120 if evaluating else None,
                      maximize=task.maximize, callbacks=[stop], verbose_eval=False)
    if refit and stop.stopped:
        raise TimeoutError("Частично обученный refit XGBoost не заменяет проверенный результат")
    best_iterations = min(model.num_boosted_rounds(),
                          int(model.best_iteration)+1 if evaluating and hasattr(model, "best_iteration")
                          else model.num_boosted_rounds())
    search_best_score = float(model.best_score) if evaluating and hasattr(model, "best_score") else None
    # Native train returns last booster after early stop; freeze the selected prefix.
    if best_iterations < model.num_boosted_rounds():
        model = model[:best_iterations]
    trained = time.monotonic()

    def predict(frame, matrix=None):
        if frame is None or not len(frame):
            return np.empty((0, len(task.classes))) if task.classification else np.empty(0)
        data = matrix if matrix is not None else xgb.DMatrix(
            frame, enable_categorical=True, nthread=threads)
        result = np.asarray(model.predict(data), dtype=float)
        if task.classification:
            return normalized_probabilities(result, len(task.classes))
        result = result.reshape(-1)
        return np.maximum(result, 0) if task.metric == "rmsle" else result

    valid, predicted = predict(x_valid, dvalid), predict(x_test)
    if not np.isfinite(valid).all() or not np.isfinite(predicted).all():
        raise ValueError("XGBoost вернул неконечные предсказания")
    details = {"best_iterations": best_iterations, "n_train": len(tr),
               "training_stopped_for_budget": stop.stopped, "training_seconds": trained-started,
               "prediction_seconds": time.monotonic()-trained, "feature_count": len(x_train.columns),
               "feature_importance": model.get_score(importance_type="gain"),
               "evaluation_used_for_fit": evaluating,
               "early_stopping_metric": task.metric if evaluating else None,
               "search_best_score": search_best_score,
               "xgboost_version": xgb.__version__}
    bundle = {"builder": builder, "model": model, "task": task.to_dict(), "config": config.to_dict()}
    return bundle, valid, predicted, details
