"""Контракты задачи и безопасного языка экспериментов. Python 3.10+."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any


METRICS = {
    "rmse": False, "mae": False, "mse": False, "rmsle": False,
    "mape": False, "r2": True, "macro_f1": True, "weighted_f1": True,
    "f1": True, "accuracy": True, "balanced_accuracy": True,
    "roc_auc": True, "average_precision": True, "log_loss": False,
}
CLASSIFICATION_METRICS = set(METRICS) - {"rmse", "mae", "mse", "rmsle", "mape", "r2"}


def native(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def fingerprint(value: Any) -> str:
    body = json.dumps(native(value), sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:20]


@dataclass
class TaskSpec:
    task_path: str
    train_path: str
    test_path: str
    sample_path: str | None
    dictionary_path: str | None
    target: str
    problem_type: str
    metric: str
    id_columns: list[str]
    feature_columns: list[str]
    categorical_columns: list[str]
    output_columns: list[str]
    prediction_kind: str = "value"
    classes: list[Any] = field(default_factory=list)
    probability_columns: list[str] = field(default_factory=list)
    positive_index: int = 1
    warnings: list[str] = field(default_factory=list)
    probability_indices: list[int] = field(default_factory=list)

    @property
    def classification(self) -> bool:
        return self.problem_type == "classification"

    @property
    def maximize(self) -> bool:
        return METRICS[self.metric]

    def to_dict(self) -> dict:
        return native(asdict(self))

    @classmethod
    def from_dict(cls, data: dict) -> "TaskSpec":
        return cls(**data)


@dataclass
class ExperimentSpec:
    model: str = "catboost"
    iterations: int = 1600
    depth: int = 6
    learning_rate: float = 0.06
    regularization: float = 5.0
    leaves: int = 31
    min_child_samples: int = 30
    class_weight_power: float = 0.0
    missing_indicators: bool = False
    frequency_encoding: bool = False
    drop_time: bool = False
    target_transform: str = "identity"
    transforms: list[dict] = field(default_factory=list)
    seed: int = 42
    hypothesis: str = "Базовый бустинг"

    def validate(self, task: TaskSpec) -> "ExperimentSpec":
        if self.model not in {"catboost", "lightgbm", "linear"}:
            raise ValueError("Недопустимая модель")
        bounds = {
            "iterations": (50, 6000), "depth": (3, 9),
            "learning_rate": (0.01, 0.3), "regularization": (0.01, 100),
            "leaves": (7, 127), "min_child_samples": (10, 300),
            "class_weight_power": (0, 1), "seed": (0, 2**31-1),
        }
        integer_fields = {"iterations", "depth", "leaves", "min_child_samples", "seed"}
        for name, (low, high) in bounds.items():
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"Неверный тип {name}")
            if name in integer_fields and not isinstance(val, int):
                raise ValueError(f"Ожидается целое {name}")
            if not low <= val <= high:
                raise ValueError(f"{name} вне допустимого диапазона")
        for name in ("missing_indicators", "frequency_encoding", "drop_time"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"Ожидается bool: {name}")
        if self.target_transform not in {"identity", "log1p"}:
            raise ValueError("Недопустимое преобразование таргета")
        if task.classification and self.target_transform != "identity":
            raise ValueError("Преобразование таргета классификации запрещено")
        if not task.classification and self.class_weight_power:
            raise ValueError("Веса классов неприменимы к регрессии")
        if not isinstance(self.hypothesis, str) or len(self.hypothesis) > 1200:
            raise ValueError("Слишком длинная гипотеза")
        if not isinstance(self.transforms, list) or len(self.transforms) > 8:
            raise ValueError("Допустимо не более 8 преобразований")
        features = set(task.feature_columns)
        for transform in self.transforms:
            if not isinstance(transform, dict) or set(transform) - {"op", "columns"}:
                raise ValueError("Недопустимая схема преобразования")
            op, columns = transform.get("op"), transform.get("columns")
            if op not in {"ratio", "difference", "sum", "log1p", "missing"}:
                raise ValueError("Недопустимая операция")
            n = 2 if op in {"ratio", "difference", "sum"} else 1
            if not isinstance(columns, list) or len(columns) != n:
                raise ValueError("Неверное число аргументов операции")
            if any(not isinstance(c, str) or c not in features for c in columns):
                raise ValueError("Операция использует неизвестный или запрещённый столбец")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def key(self) -> str:
        data = self.to_dict()
        data.pop("hypothesis", None)
        unused = {"catboost": {"leaves", "min_child_samples"},
                  "lightgbm": {"depth"},
                  "linear": {"depth", "leaves", "min_child_samples", "iterations", "learning_rate"}}
        for name in unused.get(self.model, set()):
            data.pop(name, None)
        return fingerprint(data)

    @classmethod
    def from_dict(cls, data: dict, task: TaskSpec) -> "ExperimentSpec":
        if not isinstance(data, dict):
            raise ValueError("Конфигурация должна быть объектом")
        return cls(**data).validate(task)


def parse_experiment(data: dict, task: TaskSpec) -> ExperimentSpec:
    """Закрытый реестр конфигураций; стандартный LLM-DSL остаётся прежним."""
    if isinstance(data, dict) and data.get("model") in {"extra_trees", "kernel", "retrieval"}:
        from novel_models import NovelSpec
        return NovelSpec.from_dict(data, task)
    return ExperimentSpec.from_dict(data, task)
