"""Ресурсно ограниченный портфель гипотез и адаптивное продолжение поиска."""
from __future__ import annotations
from dataclasses import replace
import random
from contracts import ExperimentSpec, TaskSpec


class Research:
    def __init__(self, task: TaskSpec, profile: dict, planner, include_alternatives: bool = False):
        self.task, self.profile, self.planner = task, profile, planner
        base = ExperimentSpec()
        self.queue = [base,
            replace(base, model="lightgbm", iterations=2400, hypothesis="Быстрый leaf-wise бустинг как независимое семейство"),
            replace(base, depth=8, iterations=2400, learning_rate=.045,
                    hypothesis="Более глубокие взаимодействия при меньшем шаге"),
            replace(base, model="lightgbm", leaves=63, min_child_samples=60,
                    regularization=8, iterations=2800, learning_rate=.04,
                    hypothesis="Больше листьев с усиленной регуляризацией"),
            replace(base, depth=5, regularization=15,
                    hypothesis="Уменьшение переобучения более мелкими деревьями")]
        feature_base = base
        if (task.classification and task.metric in {"macro_f1", "f1", "balanced_accuracy"}
                and profile.get("minority_fraction", 1) < .1):
            # При бюджете важнее проверить компенсацию дисбаланса в обоих
            # семействах, чем несколько глубоких невзвешенных моделей подряд.
            feature_base = replace(base, class_weight_power=.5)
            self.queue = self.queue[:2] + [
                replace(feature_base, hypothesis="Умеренное усиление редких классов"),
                replace(feature_base, model="lightgbm", iterations=2400,
                        hypothesis="Компенсация дисбаланса в независимом семействе бустинга"),
                replace(base, class_weight_power=1,
                        hypothesis="Проверка полной компенсации дисбаланса"),
                replace(feature_base, depth=8, learning_rate=.045, iterations=2400,
                        hypothesis="Взаимодействия признаков при сохранении веса редких классов"),
                replace(feature_base, model="lightgbm", leaves=63, min_child_samples=60,
                        iterations=2400, hypothesis="Больше листьев с весами и регуляризацией"),
                replace(feature_base, depth=5, regularization=15,
                        hypothesis="Регуляризованная взвешенная модель")]
        if profile.get("missing_train"):
            self.queue.append(replace(feature_base, missing_indicators=True,
                hypothesis="Явные индикаторы отсутствия измерений"))
        if profile.get("categorical_columns"):
            self.queue.append(replace(feature_base, frequency_encoding=True,
                hypothesis="Частоты категорий дополнят категориальные статистики"))
        if profile.get("time_column"):
            self.queue.append(replace(feature_base, drop_time=True,
                hypothesis="Проверка устойчивости без признака анонимизированного периода"))
        if not task.classification and profile.get("target_summary", {}).get("nonnegative"):
            self.queue.append(replace(base, target_transform="log1p",
                hypothesis="Логарифм таргета как проверяемая альтернатива; оценка на исходной шкале"))
        if profile.get("text_columns"):
            self.queue.insert(1, replace(base, model="linear", regularization=1,
                hypothesis="TF-IDF использует содержание текстовых полей без предобученных моделей"))
        # One inexpensive, independent family. Reference-study controls opt out.
        cats = profile.get("categorical_cardinality", {})
        width = 2*max(0, profile.get("feature_count", 0)-len(cats))+sum(min(32, n+1) for n in cats.values())
        dense_bytes = 4*profile.get("train_rows", 0)*max(1, width)
        if (include_alternatives and not profile.get("text_columns")
                and profile.get("feature_count", 0) <= 150 and dense_bytes <= 512*2**20):
            from novel_models import NovelSpec
            self.queue.insert(4, NovelSpec(forest_depth=24, min_leaf=4, max_features=1.,
                class_weight_power=feature_base.class_weight_power,
                hypothesis="Независимый случайный лес: поиск дополнительного сигнала для ансамбля"))
        self.attempted = set()
        self.rng = random.Random(42)
        self.steps = 0

    def next(self, history: list[dict]) -> ExperimentSpec | None:
        self.steps += 1
        if self.steps > 2 and self.steps % 3 == 0 and self.planner.available:
            proposed = self.planner.propose(self.task, self.profile, history)
            if proposed and proposed.key not in self.attempted:
                self.attempted.add(proposed.key)
                return proposed
        while self.queue:
            candidate = self.queue.pop(0)
            if candidate.key not in self.attempted:
                self.attempted.add(candidate.key)
                return candidate.validate(self.task)
        successes = [r for r in history if r.get("status") == "ok"
                     and r.get("config", {}).get("model") in {"catboost", "lightgbm", "linear"}]
        if not successes:
            return None
        successes.sort(key=lambda r: r["score"], reverse=self.task.maximize)
        base = ExperimentSpec.from_dict(successes[0]["config"], self.task)
        for _ in range(20):
            candidate = replace(base, depth=self.rng.choice([5, 6, 7, 8]),
                regularization=self.rng.choice([2., 5., 10., 20.]),
                learning_rate=self.rng.choice([.035, .05, .075]),
                leaves=self.rng.choice([15, 31, 63]),
                hypothesis="Уточнение регуляризации и шага вокруг лучшей конфигурации")
            if candidate.key not in self.attempted:
                self.attempted.add(candidate.key)
                return candidate.validate(self.task)
        return None


def choose_finalists(task, successes, y_valid, prediction_loader):
    """Ищем дополнение к лидеру по метрике смеси, а не по личному месту модели."""
    from metrics import better, evaluate
    ranked = sorted(successes, key=lambda r: r["score"], reverse=task.maximize)
    if not ranked:
        raise ValueError("Нет успешных моделей")
    leader = ranked[0]
    primary = prediction_loader(leader)
    complements = {}
    for candidate in ranked[1:]:
        other = prediction_loader(candidate)
        best_score, selected_weight = leader["score"], None
        for weight in (.25, .5, .75):
            score = evaluate(task, y_valid, weight*primary+(1-weight)*other)[task.metric]
            if better(task, score, best_score, tolerance=.001):
                best_score, selected_weight = score, weight
        if selected_weight is not None:
            family = candidate["config"]["model"]
            previous = complements.get(family)
            if previous is None or better(task, best_score, previous["score"]):
                complements[family] = {"result": candidate, "weight": selected_weight, "score": best_score}
    ordered = sorted(complements.values(), key=lambda r: r["score"], reverse=task.maximize)[:2]
    if not ordered:
        return ranked[:2], {}
    return ([leader]+[v["result"] for v in ordered],
            {v["result"]["config_key"]: v["weight"] for v in ordered})
