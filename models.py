"""Обучение моделей. Все тяжёлые операции выполняются в дочернем процессе."""
from __future__ import annotations
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier

from contracts import ExperimentSpec, TaskSpec
from preprocessing import FeatureBuilder
from metrics import normalized_probabilities


class CatBoostDeadline:
    def __init__(self, deadline: float):
        self.deadline = deadline
        self.stopped = False

    def after_iteration(self, info):
        self.stopped = time.monotonic() >= self.deadline
        return not self.stopped


def sample_weights(y: np.ndarray, classes: int, power: float) -> np.ndarray | None:
    if power <= 0:
        return None
    counts = np.bincount(y, minlength=classes).astype(float)
    weights = (len(y) / (classes * np.maximum(counts, 1))) ** power
    result = weights[y] / np.mean(weights[y])
    return None if np.allclose(result, 1) else result


def linear_model(task: TaskSpec, config: ExperimentSpec, frame: pd.DataFrame, profile: dict):
    text = [c for c in profile.get("text_columns", []) if c in frame][:2]
    categorical = [c for c in frame if not pd.api.types.is_numeric_dtype(frame[c]) and c not in text]
    numeric = [c for c in frame if c not in categorical and c not in text]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("fill", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler(with_mean=False))]), numeric))
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="infrequent_if_exist",
                             min_frequency=3, max_categories=128, sparse_output=True), categorical))
    for i, col in enumerate(text):
        transformers.append((f"text_{i}", TfidfVectorizer(max_features=30000,
                             ngram_range=(1, 2), min_df=1 if len(frame) < 50 else 2,
                             sublinear_tf=True), col))
    encoder = ColumnTransformer(transformers, sparse_threshold=1.0)
    if task.classification:
        estimator = LogisticRegression(C=1/config.regularization, solver="liblinear",
                                        max_iter=400, random_state=config.seed)
    else:
        estimator = Ridge(alpha=config.regularization, solver="lsqr")
    return Pipeline([("features", encoder), ("estimator", estimator)])


def train_model(task: TaskSpec, config: ExperimentSpec, profile: dict, train: pd.DataFrame,
                test: pd.DataFrame, y: np.ndarray, train_indices: np.ndarray,
                valid_indices: np.ndarray, threads: int, refit: bool = False,
                fixed_iterations: int | None = None,
                time_limit: float | None = None) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    started = time.monotonic()
    deadline = started + time_limit if time_limit is not None else float("inf")
    budget_stopped = False
    builder = FeatureBuilder(task, config, profile).fit(train.iloc[train_indices])
    lightgbm = config.model == "lightgbm"
    x_train = builder.transform(train.iloc[train_indices], categorical_codes=lightgbm)
    x_valid = builder.transform(train.iloc[valid_indices], categorical_codes=lightgbm) if len(valid_indices) else None
    x_test = builder.transform(test, categorical_codes=lightgbm)
    y_train = y[train_indices].copy()
    y_valid = y[valid_indices].copy()
    if config.target_transform == "log1p":
        if np.min(y_train) < 0:
            raise ValueError("log1p таргета требует неотрицательных значений")
        y_train = np.log1p(y_train)
        y_valid = np.log1p(np.maximum(y_valid, 0))
    iterations = int(fixed_iterations or config.iterations)
    weights = sample_weights(y_train, len(task.classes), config.class_weight_power) if task.classification else None
    categorical = [c for c in builder.categorical if c in x_train]
    if task.classification and len(np.unique(y_train)) < 2:
        model = DummyClassifier(strategy="prior").fit(x_train, y_train)
        best_iterations = 1
    elif config.model == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        params = dict(iterations=iterations, depth=config.depth, learning_rate=config.learning_rate,
                      l2_leaf_reg=config.regularization, random_seed=config.seed,
                      thread_count=threads, allow_writing_files=False, verbose=False,
                      task_type="CPU")
        if task.classification:
            params["loss_function"] = "Logloss" if len(task.classes) == 2 else "MultiClass"
            if len(task.classes) > 2:
                params["classes_count"] = len(task.classes)
            if task.metric in {"macro_f1", "weighted_f1"}:
                average = "Macro" if task.metric == "macro_f1" else "Weighted"
                params["eval_metric"] = f"TotalF1:average={average}" + (";use_weights=false" if weights is not None else "")
            elif task.metric == "accuracy":
                params["eval_metric"] = "Accuracy" + (";use_weights=false" if weights is not None else "")
            model = CatBoostClassifier(**params)
        else:
            params["loss_function"] = "MAE" if task.metric == "mae" else "RMSE"
            model = CatBoostRegressor(**params)
        fit = dict(cat_features=categorical, sample_weight=weights)
        stop = CatBoostDeadline(deadline)
        fit["callbacks"] = [stop]
        if x_valid is not None and not refit:
            fit.update(eval_set=(x_valid, y_valid), early_stopping_rounds=160, use_best_model=True)
        model.fit(x_train, y_train, **fit)
        budget_stopped = stop.stopped
        best_iterations = int(model.tree_count_)
    elif lightgbm:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(task.task_path).parent / ".alfa_runs" / "matplotlib-cache"))
        import lightgbm as lgb
        params = dict(n_estimators=iterations, learning_rate=config.learning_rate,
                      num_leaves=config.leaves, min_child_samples=config.min_child_samples,
                      reg_lambda=config.regularization, colsample_bytree=.9,
                      subsample=.85, subsample_freq=1, random_state=config.seed,
                      n_jobs=threads, verbosity=-1, deterministic=True, force_col_wise=True)
        if task.classification:
            model = lgb.LGBMClassifier(**params)
        else:
            params["objective"] = "regression_l1" if task.metric == "mae" else "regression"
            model = lgb.LGBMRegressor(**params)
        fit = dict(sample_weight=weights)
        time_stopped = [False]

        def time_callback(environment):
            if time.monotonic() >= deadline:
                time_stopped[0] = True
                if environment.evaluation_result_list:
                    raise lgb.callback.EarlyStopException(environment.iteration, environment.evaluation_result_list)
                raise TimeoutError("Не завершён refit в выделенное время")

        time_callback.order = 5
        callbacks = [time_callback]
        if x_valid is not None and not refit:
            callbacks.append(lgb.early_stopping(120, verbose=False))
            fit.update(eval_set=[(x_valid, y_valid)])
        fit["callbacks"] = callbacks
        model.fit(x_train, y_train, **fit)
        budget_stopped = time_stopped[0]
        best_iterations = int(model.best_iteration_ or iterations)
    elif config.model == "linear":
        model = linear_model(task, config, x_train, profile)
        fit = {"estimator__sample_weight": weights} if weights is not None else {}
        model.fit(x_train, y_train, **fit)
        best_iterations = 1
    else:
        raise ValueError("Неизвестная модель")
    if refit and budget_stopped:
        raise TimeoutError("Частично обученный refit не заменяет проверенную модель")
    training_seconds = time.monotonic() - started

    def predict(frame):
        if frame is None:
            return np.empty((0, len(task.classes))) if task.classification else np.empty(0)
        if task.classification:
            raw = np.asarray(model.predict_proba(frame), dtype=float)
            probability = np.zeros((len(frame), len(task.classes)))
            for column, label in enumerate(model.classes_):
                probability[:, int(label)] = raw[:, column]
            return normalized_probabilities(probability, len(task.classes))
        values = np.asarray(model.predict(frame), dtype=float).reshape(-1)
        if config.target_transform == "log1p":
            values = np.expm1(np.clip(values, -50, 50))
        if task.metric == "rmsle":
            values = np.maximum(values, 0)
        return values

    valid_pred, test_pred = predict(x_valid), predict(x_test)
    prediction_seconds = time.monotonic() - started - training_seconds
    importance = {}
    if hasattr(model, "feature_importances_"):
        importance = {str(c): float(v) for c, v in sorted(zip(x_train.columns, model.feature_importances_),
                                                       key=lambda pair: -pair[1])[:25]}
    bundle = {"builder": builder, "model": model, "task": task.to_dict(), "config": config.to_dict()}
    details = {"best_iterations": best_iterations, "n_train": len(train_indices),
               "training_stopped_for_budget": budget_stopped,
               "training_seconds": training_seconds, "prediction_seconds": prediction_seconds,
               "feature_count": len(x_train.columns), "feature_importance": importance}
    return bundle, valid_pred, test_pred, details
