"""Ресурсно ограниченный портфель гипотез и адаптивное продолжение поиска."""
from __future__ import annotations
from dataclasses import replace
import random
from contracts import ExperimentSpec, TaskSpec


class Research:
    def __init__(self, task: TaskSpec, profile: dict, planner, include_alternatives: bool = False,
                 history_schema: dict | None = None, operator_options: tuple = ()):
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
                        leaves=63, min_child_samples=60,
                        hypothesis="Контрольный LightGBM с умеренными весами редких классов"),
                replace(base, class_weight_power=1,
                        hypothesis="Проверка полной компенсации дисбаланса"),
                replace(feature_base, depth=8, learning_rate=.045, iterations=2400,
                        hypothesis="Взаимодействия признаков при сохранении веса редких классов"),
                replace(feature_base, model="lightgbm", leaves=63, min_child_samples=60,
                        iterations=2400, hypothesis="Больше листьев с весами и регуляризацией"),
                replace(feature_base, depth=5, regularization=15,
                        hypothesis="Регуляризованная взвешенная модель")]
            self.queue.insert(5, replace(feature_base, model="lightgbm", iterations=2400,
                hypothesis="Дополнительный контроль с прежним числом листьев"))
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
        if history_schema and not task.classification:
            from history_router import HistoryRouterSpec
            routes = [HistoryRouterSpec(history_schema=history_schema)]
            if "robust_fallback" in operator_options:
                routes.append(HistoryRouterSpec(history_schema=history_schema, fallback="forest_huber",
                    fallback_weight=.5, fallback_iterations=2400,
                    hypothesis="История для известных ключей, смесь леса и Huber для неизвестных"))
            if "median" in operator_options:
                routes.append(HistoryRouterSpec(history_schema=history_schema, history_policy="median",
                    hypothesis="Историческая ветвь с train-only медианами и индикаторами"))
            if "seed_average" in operator_options:
                routes.append(HistoryRouterSpec(history_schema=history_schema, seed_count=2,
                    hypothesis="Среднее двух заранее заданных исторических маршрутизаторов"))
            self.queue[5:5] = routes
        # Дешёвый самостоятельный контроль обязателен и для числовых таблиц,
        # а не только для текста или удачного предложения LLM.
        linear = replace(feature_base, model="linear",
            regularization=1 if profile.get("text_columns") else 5,
            hypothesis="Обязательная ранняя линейная модель с train-only масштабированием")
        self.queue.insert(0, linear)
        self.required_keys = {linear.key, base.key,
            replace(base, model="lightgbm", iterations=2400).key}
        self.required_keys.update(c.key for c in self.queue
                                  if c.model in {"history_router", "extra_trees"})
        if (task.classification and task.metric == "macro_f1"
                and profile.get("minority_fraction", 1) < .1):
            self.required_keys.update({replace(base,class_weight_power=.5).key,
                replace(base,model="lightgbm",iterations=2400,leaves=63,
                        min_child_samples=60,class_weight_power=.5).key})
        self.attempted = set()
        self.rng = random.Random(42)
        self.steps = 0

    @property
    def essential_pending(self):
        return not self.required_keys.issubset(self.attempted)

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
            if base.model == "linear":
                weight = (self.rng.choice([0., .5, 1.]) if self.task.classification
                          and self.task.metric in {"macro_f1", "f1", "balanced_accuracy"}
                          else base.class_weight_power)
                candidate = replace(base, regularization=self.rng.choice([.01,.1,1.,5.,10.,100.]),
                    class_weight_power=weight,
                    hypothesis="Уточнение регуляризации самостоятельной линейной модели")
            else:
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
        if leader["config"]["model"] == "history_router" and candidate["config"]["model"] == "history_router":
            # Параметры одного gate выбираются как рецепты; смешивание одинаковых
            # исторических ветвей создало бы незаявленную сетку весов резерва.
            continue
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


def search_plan(task, successes, y_valid, prediction_loader):
    """Зафиксировать весь состав и веса на поиске, до доступа к confirm."""
    from metrics import better, evaluate
    candidates, blend_weights = choose_finalists(task, successes, y_valid, prediction_loader)
    leader = candidates[0]
    best = {"members": [leader["config_key"]], "weights": [1.], "score": leader["score"]}
    primary = prediction_loader(leader)
    for candidate in candidates[1:]:
        weight = blend_weights.get(candidate["config_key"])
        if weight is None:
            continue
        score = evaluate(task, y_valid, weight*primary+(1-weight)*prediction_loader(candidate))[task.metric]
        if better(task, score, best["score"], .001):
            best = {"members": [leader["config_key"], candidate["config_key"]],
                    "weights": [weight, 1-weight], "score": score}
    return best


def cap_iterations(config, maximum):
    """Локальный cap корректен и для рецепта с вложенной базовой моделью."""
    if not maximum:
        return config
    if config.model == "proposal":
        base = dict(config.base_config)
        base["iterations"] = min(base.get("iterations", maximum), maximum)
        return replace(config, base_config=base)
    if config.model == "history_router":
        return replace(config, iterations=min(config.iterations, maximum),
                       fallback_iterations=min(config.fallback_iterations, maximum))
    return replace(config, iterations=min(config.iterations, maximum))


def choose_confirmed_plan(task, planned, legacy_planned, legacy_control, score_loader,
                          oof_selected=False):
    """Сначала прежнее правило fallback, затем OOF против его результата.

    Все переданные планы зафиксированы до confirm. Неполная проверка
    альтернативы не вытесняет старый план, даже если её отдельный score хорош.
    """
    from metrics import better
    legacy_score, control_score = score_loader(legacy_planned), score_loader(legacy_control)
    guard = legacy_planned
    if control_score is not None and (legacy_score is None or
            (legacy_planned != legacy_control and not better(task, legacy_score, control_score, .001))):
        guard = legacy_control
    guard_score = score_loader(guard)
    proposed_score = score_loader(planned)
    chosen = guard
    if oof_selected and guard_score is not None and proposed_score is not None:
        if better(task, proposed_score, guard_score, .001):
            chosen = planned
    return chosen, {"planned_score": proposed_score, "legacy_score": legacy_score,
                    "baseline_score": control_score, "guard_score": guard_score}


def control_plan(task, profile, successes, y_valid, prediction_loader):
    """Известный безопасный контроль задаётся до confirm по общим условиям задачи."""
    from metrics import evaluate
    if not task.classification:
        forest = next((r for r in successes if r["config"]["model"] == "extra_trees"), None)
        if forest:
            return {"members": [forest["config_key"]], "weights": [1.], "score": forest["score"]}
    if task.classification and task.metric == "macro_f1" and profile.get("minority_fraction", 1) < .1:
        expected_cb = ExperimentSpec(class_weight_power=.5).key
        expected_lgb = ExperimentSpec(model="lightgbm",iterations=2400,leaves=63,
                                      min_child_samples=60,class_weight_power=.5).key
        cb = next((r for r in successes if r["config"]["model"] == "catboost"
            and ExperimentSpec.from_dict(r["config"],task).key == expected_cb), None)
        lgb = next((r for r in successes if r["config"]["model"] == "lightgbm"
            and ExperimentSpec.from_dict(r["config"],task).key == expected_lgb), None)
        if cb and lgb:
            score = evaluate(task, y_valid, .5*prediction_loader(lgb)+.5*prediction_loader(cb))[task.metric]
            return {"members": [lgb["config_key"], cb["config_key"]], "weights": [.5, .5], "score": score}
    return search_plan(task, successes, y_valid, prediction_loader)
