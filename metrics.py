"""Единая невзвешенная оценка на исходной шкале таргета."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, log_loss, mean_absolute_percentage_error, r2_score, roc_auc_score)
from contracts import TaskSpec


def normalized_probabilities(prediction: np.ndarray, classes: int) -> np.ndarray:
    pred = np.asarray(prediction, dtype=float)
    if pred.ndim == 1 and classes == 2:
        pred = np.column_stack([1 - pred, pred])
    if pred.ndim != 2 or pred.shape[1] != classes or not np.isfinite(pred).all():
        raise ValueError("Некорректная матрица вероятностей")
    if (pred < -1e-7).any() or (pred > 1+1e-7).any():
        raise ValueError("Вероятности вне [0,1]")
    pred = np.clip(pred, 0, 1)
    sums = pred.sum(axis=1, keepdims=True)
    if (sums <= 0).any():
        raise ValueError("Нулевая сумма вероятностей")
    return pred / sums


def evaluate(task: TaskSpec, y: np.ndarray, prediction: np.ndarray) -> dict:
    if not len(y):
        raise ValueError("Пустой набор для оценки")
    if task.classification:
        p = normalized_probabilities(prediction, len(task.classes))
        labels = p.argmax(axis=1)
        metrics = {
            "macro_f1": float(f1_score(y, labels, labels=np.arange(len(task.classes)), average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, labels, average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(y, labels)),
            "balanced_accuracy": float(balanced_accuracy_score(y, labels)),
            "log_loss": float(log_loss(y, p, labels=np.arange(len(task.classes)))),
        }
        if len(task.classes) == 2:
            positive = task.positive_index
            binary_y = (y == positive).astype(int)
            metrics["f1"] = float(f1_score(binary_y, labels == positive, zero_division=0))
            if len(np.unique(binary_y)) == 2 and task.metric in {"roc_auc", "average_precision"}:
                metrics["roc_auc"] = float(roc_auc_score(binary_y, p[:, positive]))
                metrics["average_precision"] = float(average_precision_score(binary_y, p[:, positive]))
        elif task.metric == "roc_auc" and len(np.unique(y)) == len(task.classes):
            metrics["roc_auc"] = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
        per_class = f1_score(y, labels, labels=np.arange(len(task.classes)), average=None, zero_division=0)
        metrics["per_class_f1"] = {str(c): float(v) for c, v in zip(task.classes, per_class)}
    else:
        pred = np.asarray(prediction, dtype=float).reshape(-1)
        if len(pred) != len(y) or not np.isfinite(pred).all():
            raise ValueError("Некорректные предсказания регрессии")
        difference = pred - y
        metrics = {"rmse": float(np.sqrt(np.mean(difference**2))),
                   "mse": float(np.mean(difference**2)), "mae": float(np.mean(np.abs(difference))),
                   "r2": float(r2_score(y, pred)),
                   "mape": float(mean_absolute_percentage_error(y, pred))}
        if np.min(y) >= 0:
            metrics["rmsle"] = float(np.sqrt(np.mean((np.log1p(np.maximum(pred, 0))-np.log1p(y))**2)))
    metrics = {key: value for key, value in metrics.items()
               if isinstance(value, dict) or np.isfinite(value)}
    if task.metric not in metrics or not np.isfinite(metrics[task.metric]):
        raise ValueError(f"Нельзя вычислить метрику {task.metric} на данном разбиении")
    return metrics


def better(task: TaskSpec, score: float, previous: float | None, tolerance: float = 1e-8) -> bool:
    if not np.isfinite(score):
        return False
    if previous is None:
        return True
    gain = score - previous if task.maximize else previous - score
    return gain > tolerance * max(1.0, abs(previous))
