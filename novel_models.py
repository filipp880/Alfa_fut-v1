"""Три альтернативных принципа обучения, без внешних данных и весов."""
from dataclasses import dataclass, fields
import math
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from contracts import ExperimentSpec, fingerprint
from metrics import normalized_probabilities
from models import sample_weights


@dataclass
class NovelSpec(ExperimentSpec):
    model: str = "extra_trees"
    forest_trees: int = 256
    forest_depth: int = 28
    min_leaf: int = 1
    max_features: float = .7
    components: int = 384
    kernel_scale: float = .5
    neighbors: int = 25
    metric_features: int = 24

    def validate(self, task):
        if self.model not in {"extra_trees", "kernel", "retrieval"}:
            raise ValueError("Недопустимое исследовательское семейство")
        standard = {f.name: getattr(self, f.name) for f in fields(ExperimentSpec)}
        standard["model"] = "lightgbm"
        ExperimentSpec.from_dict(standard, task)
        if (self.transforms or self.target_transform != "identity" or self.drop_time
                or self.missing_indicators or self.frequency_encoding):
            raise ValueError("У альтернативного семейства фиксированная предобработка")
        bounds = {"forest_trees": (16, 512), "forest_depth": (4, 32), "min_leaf": (1, 100),
                  "max_features": (.1, 1), "components": (16, 1024), "kernel_scale": (.1, 20),
                  "neighbors": (1, 150), "metric_features": (4, 64)}
        real = {"max_features", "kernel_scale"}
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not low <= value <= high
                    or (name not in real and not isinstance(value, int))):
                raise ValueError(f"Недопустимый параметр {name}")
        return self

    @property
    def key(self):
        names = {"extra_trees": ["forest_trees", "forest_depth", "min_leaf", "max_features"],
                 "kernel": ["components", "kernel_scale", "regularization"],
                 "retrieval": ["neighbors", "metric_features"]}[self.model]
        return fingerprint({k: getattr(self, k) for k in ["model", "seed", "class_weight_power", *names]})


class MixedEncoder:
    """Все медианы, ранги и словари категорий обучаются только на fit-строках."""
    def fit_transform(self, task, frame):
        self.features = list(task.feature_columns)
        cats = [c for c in self.features if c in task.categorical_columns]
        nums = [c for c in self.features if c not in cats]
        self.cats, self.nums = cats, nums
        numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                                QuantileTransformer(n_quantiles=min(256, len(frame)),
                                    output_distribution="normal", subsample=10000, random_state=42))
        self.encoder = ColumnTransformer([
            ("numeric", numeric, nums),
            ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=5,
                                          max_categories=32, sparse_output=False, dtype=np.float32), cats),
        ], sparse_threshold=0)
        return np.ascontiguousarray(self.encoder.fit_transform(self.clean(frame)), dtype=np.float32)

    def clean(self, frame):
        result = frame[self.features].copy()
        for col in self.cats:
            result[col] = result[col].fillna("__MISSING__").astype(str)
        for col in self.nums:
            result[col] = pd.to_numeric(result[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return result

    def transform(self, frame):
        return np.ascontiguousarray(self.encoder.transform(self.clean(frame)), dtype=np.float32)


class AlternativeModel:
    def __init__(self, task, config, threads):
        self.task, self.config, self.threads = task, config, threads

    def fit(self, frame, y):
        task, c = self.task, self.config
        c.validate(task)
        self.encoder = MixedEncoder()
        x = self.encoder.fit_transform(task, frame)
        self.feature_count = x.shape[1]
        weights = sample_weights(y, len(task.classes), c.class_weight_power) if task.classification else None
        if c.model == "extra_trees":
            estimator = ExtraTreesClassifier if task.classification else ExtraTreesRegressor
            self.model = estimator(n_estimators=c.forest_trees, max_depth=c.forest_depth,
                        min_samples_leaf=c.min_leaf, max_features=c.max_features,
                        n_jobs=self.threads, random_state=c.seed)
            self.model.fit(x, y, sample_weight=weights)
        elif c.model == "kernel":
            rng = np.random.default_rng(c.seed)
            a, b = rng.integers(len(x), size=(2, min(2048, len(x)*2)))
            distances = np.sum((x[a]-x[b])**2, axis=1)
            median = float(np.median(distances[distances > 1e-8])) if (distances > 1e-8).any() else 1.
            self.kernel = Nystroem(kernel="rbf", gamma=c.kernel_scale/median,
                                  n_components=min(c.components, len(x)), random_state=c.seed,
                                  n_jobs=self.threads)
            z = self.kernel.fit_transform(x).astype(np.float32)
            z = np.concatenate([z, x/np.sqrt(max(1, x.shape[1]))], axis=1)
            if task.classification:
                self.model = LogisticRegression(C=1/c.regularization, max_iter=300,
                                               tol=1e-3, solver="lbfgs", random_state=c.seed)
            else:
                self.model = Ridge(alpha=c.regularization, solver="cholesky")
            self.model.fit(z, y, sample_weight=weights)
        else:
            # Supervised feature weighting learned only from training labels.
            selector_type = ExtraTreesClassifier if task.classification else ExtraTreesRegressor
            selector = selector_type(n_estimators=48, max_depth=12, min_samples_leaf=10,
                                     max_features=.7, n_jobs=self.threads, random_state=c.seed)
            selector.fit(x, y, sample_weight=weights)
            importance = np.asarray(selector.feature_importances_)
            self.columns = np.argsort(-importance, kind="stable")[:min(c.metric_features, x.shape[1])]
            selected = np.maximum(importance[self.columns], 1e-6)
            self.scale = np.clip(np.sqrt(selected/selected.mean()), .25, 3).astype(np.float32)
            self.memory_x = np.ascontiguousarray(x[:, self.columns]*self.scale)
            self.memory_y = np.array(y, copy=True)
            self.neighbor_index = NearestNeighbors(n_neighbors=min(c.neighbors, len(y)),
                                                 algorithm="brute", metric="euclidean", n_jobs=1)
            self.neighbor_index.fit(self.memory_x)
            if task.classification:
                counts = np.bincount(y, minlength=len(task.classes)).astype(float)
                self.prior = counts/counts.sum()
                self.class_correction = (len(y)/(len(counts)*np.maximum(counts, 1)))**c.class_weight_power
        return self

    def predict(self, frame):
        task, c = self.task, self.config
        if not len(frame):
            return np.empty((0, len(task.classes))) if task.classification else np.empty(0)
        x = self.encoder.transform(frame)
        if c.model == "kernel":
            z = self.kernel.transform(x).astype(np.float32)
            x = np.concatenate([z, x/np.sqrt(max(1, x.shape[1]))], axis=1)
        if c.model != "retrieval":
            if not task.classification:
                result = np.asarray(self.model.predict(x), dtype=float)
            else:
                result = np.zeros((len(x), len(task.classes)))
                raw = self.model.predict_proba(x)
                result[:, self.model.classes_.astype(int)] = raw
        else:
            queries = np.ascontiguousarray(x[:, self.columns]*self.scale)
            result = np.zeros((len(x), len(task.classes))) if task.classification else np.zeros(len(x))
            for start in range(0, len(queries), 512):
                distance, indices = self.neighbor_index.kneighbors(queries[start:start+512])
                zero = distance < 1e-7
                weight = 1/np.maximum(distance, 1e-3)**2
                weight = np.where(zero.any(axis=1)[:, None], zero.astype(float), weight)
                weight /= weight.sum(axis=1, keepdims=True)
                labels = self.memory_y[indices]
                if task.classification:
                    scores = np.stack([(weight*(labels == cl)).sum(axis=1)
                                       for cl in range(len(task.classes))], axis=1)
                    result[start:start+len(indices)] = (scores+.001*self.prior)*self.class_correction
                else:
                    result[start:start+len(indices)] = (weight*labels).sum(axis=1)
        if task.classification:
            result /= result.sum(axis=1, keepdims=True)
            return normalized_probabilities(result, len(task.classes))
        return np.maximum(result, 0) if task.metric == "rmsle" else result


def portfolio(task, profile):
    from dataclasses import replace
    power = .5 if task.classification and task.metric in {"macro_f1", "f1", "balanced_accuracy"} else 0.
    base = NovelSpec(class_weight_power=power)
    return {
        "extra_trees": [base, replace(base, min_leaf=4, forest_depth=24, max_features=1.),
                        replace(base, min_leaf=12, forest_depth=20)],
        "kernel": [replace(base, model="kernel", components=384, kernel_scale=.5, regularization=1.),
                   replace(base, model="kernel", components=768, kernel_scale=2., regularization=3.),
                   replace(base, model="kernel", components=768, kernel_scale=8., regularization=10.)],
        "retrieval": [replace(base, model="retrieval", neighbors=5, metric_features=16),
                      replace(base, model="retrieval", neighbors=25, metric_features=24),
                      replace(base, model="retrieval", neighbors=75, metric_features=32)],
    }
