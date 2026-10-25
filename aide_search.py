"""Ограниченный поиск по дереву безопасных конфигураций, вдохновлённый AIDE.

Это исследовательская абляция выбора родителя: у linear и tree один пул
атомарных действий и одинаковый лимит попыток. Генерации Python-кода нет.
Поиск видит только поисковую метрику; подтверждение остаётся снаружи.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
import random

from contracts import ExperimentSpec


ACTION_FIELDS = frozenset({"iterations", "depth", "learning_rate", "regularization",
                           "leaves", "min_child_samples", "class_weight_power",
                           "missing_indicators", "frequency_encoding", "drop_time",
                           "target_transform"})


def default_actions(task, base_config):
    """Общий заранее известный пул; не зависит от подтверждающих меток."""
    base = base_config if isinstance(base_config, ExperimentSpec) else ExperimentSpec.from_dict(base_config, task)
    actions = [{"field": "regularization", "value": v} for v in (2., 10., 20.)]
    actions += [{"field": "learning_rate", "value": v} for v in (.035, .05, .075)]
    actions += [{"field": "iterations", "value": v} for v in (800, 1600, 2400)]
    if base.model == "catboost":
        actions += [{"field": "depth", "value": v} for v in (5, 7, 8)]
    elif base.model == "lightgbm":
        actions += [{"field": "leaves", "value": v} for v in (15, 63)]
        actions += [{"field": "min_child_samples", "value": v} for v in (60, 100)]
    actions += [{"field": name, "value": True} for name in
                ("missing_indicators", "frequency_encoding", "drop_time")]
    if task.classification:
        actions += [{"field": "class_weight_power", "value": v} for v in (0., .5, 1.)]
    return validate_actions(task, base, actions)


def validate_actions(task, base_config, actions):
    """Закрытая проверка LLM-пула через тот же контракт, что у моделей."""
    base = base_config if isinstance(base_config, ExperimentSpec) else ExperimentSpec.from_dict(base_config, task)
    if base.model not in {"catboost", "lightgbm"}:
        raise ValueError("Для абляции разрешены только стандартные бустинги")
    if not isinstance(actions, list) or not 1 <= len(actions) <= 40:
        raise ValueError("Допустимо от 1 до 40 атомарных действий")
    cleaned, seen = [], set()
    for action in actions:
        if not isinstance(action, dict) or set(action) != {"field", "value"}:
            raise ValueError("Действие должно содержать только field/value")
        field, value = action["field"], action["value"]
        if not isinstance(field, str) or field not in ACTION_FIELDS or not isinstance(value, (str, int, float, bool)):
            raise ValueError("Недопустимое поле либо тип атомарного действия")
        replace(base, **{field: value}).validate(task)
        key = (field, type(value).__name__, value)
        if key not in seen:
            seen.add(key)
            cleaned.append({"field": field, "value": value})
    return cleaned


class BoundedSearch:
    """Одна текущая pending-попытка, воспроизводимые родители и полный журнал.

    seed управляет только выбором/порядком поиска, seed модели не меняется.
    У обеих политик одинаковое исправление ресурсоёмкой неуспешной попытки:
    не более одного дочернего repair-узла, только действием из общего пула.
    """
    def __init__(self, task, base_config, base_result, policy, actions=None, seed=42,
                 max_attempts=6, explore_probability=.2):
        if policy not in {"linear", "tree"}:
            raise ValueError("Политика должна быть linear или tree")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise ValueError("Некорректный seed поиска")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 12:
            raise ValueError("Допустимо от 1 до 12 попыток")
        if isinstance(explore_probability, bool) or not isinstance(explore_probability, (int, float)) or not 0 <= explore_probability <= 1:
            raise ValueError("Некорректная вероятность исследования")
        self.task, self.policy, self.seed = task, policy, seed
        self.max_attempts = max_attempts
        self.explore_probability = float(explore_probability)
        self.base = base_config if isinstance(base_config, ExperimentSpec) else ExperimentSpec.from_dict(base_config, task)
        self.base.validate(task)
        if base_result.get("status") != "ok" or not self._finite_score(base_result.get("score")):
            raise ValueError("Нужен успешный общий исходный результат")
        if "config_key" in base_result and base_result["config_key"] != self.base.key:
            raise ValueError("Исходный результат относится к другой конфигурации")
        self.actions = validate_actions(task, self.base,
                                        default_actions(task, self.base) if actions is None else actions)
        self.rng = random.Random(seed)
        self.action_order = list(range(len(self.actions)))
        self.rng.shuffle(self.action_order)
        self.nodes = [{"id": 0, "parent": None, "children": [], "stage": "shared_root",
                       "action": None, "config": self.base.to_dict(), "config_key": self.base.key,
                       "status": "ok", "score": float(base_result["score"]),
                       "wall_seconds": 0., "shared_training_seconds": float(base_result.get("wall_seconds", 0)),
                       "error_type": None, "error": None,
                       "directory": base_result.get("directory")}]
        self.attempted = {self.base.key}
        self.pending = None
        self.repaired = set()
        self.exhausted = False

    @staticmethod
    def _finite_score(value):
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)

    def _rank(self, node):
        return node["score"] if self.task.maximize else -node["score"]

    def _candidates(self, parent, repair=False):
        config = ExperimentSpec.from_dict(parent["config"], self.task)
        for index in self.action_order:
            action = self.actions[index]
            if repair:
                # Ремонт конфигурации ограничен явным снижением ресурсоёмкости.
                field, value = action["field"], action["value"]
                if field not in {"iterations", "depth", "leaves"} or value >= getattr(config, field):
                    continue
            candidate = replace(config, **{action["field"]: action["value"]},
                hypothesis=f"{self.policy}: {'исправление' if repair else 'атомарная проверка'} "
                           f"{action['field']}={action['value']}").validate(self.task)
            if candidate.key not in self.attempted:
                yield candidate, action

    def next(self):
        if self.pending is not None:
            raise RuntimeError("Сначала сохраните результат текущей попытки")
        if len(self.nodes)-1 >= self.max_attempts:
            return None
        last = self.nodes[-1]
        selected = None
        if last["status"] != "ok" and last["stage"] != "repair" and last["id"] not in self.repaired:
            self.repaired.add(last["id"])
            candidate = next(self._candidates(last, repair=True), None)
            if candidate is not None:
                selected = last, *candidate, "repair"
        if selected is None:
            good = [node for node in self.nodes if node["status"] == "ok"]
            if self.policy == "linear":
                parents = [good[-1]]
            else:
                ordered = sorted(good, key=self._rank, reverse=True)
                if len(ordered) > 1 and self.rng.random() < self.explore_probability:
                    # Возврат к недоисследованной успешной ветви.
                    explored = min(ordered, key=lambda n: (len(n["children"]), -self._rank(n), n["id"]))
                    ordered = [explored]+[n for n in ordered if n["id"] != explored["id"]]
                parents = ordered
            for parent in parents:
                candidate = next(self._candidates(parent), None)
                if candidate is not None:
                    selected = parent, *candidate, "improve"
                    break
        if selected is None:
            self.exhausted = True
            return None
        parent, candidate, action, stage = selected
        node = {"id": len(self.nodes), "parent": parent["id"], "children": [], "stage": stage,
                "action": deepcopy(action), "config": candidate.to_dict(), "config_key": candidate.key,
                "status": "pending", "score": None, "wall_seconds": None,
                "error_type": None, "error": None, "directory": None}
        parent["children"].append(node["id"])
        self.nodes.append(node)
        self.attempted.add(candidate.key)
        self.pending = node["id"]
        return candidate

    def observe(self, result):
        if self.pending is None:
            raise RuntimeError("Нет ожидающей результата попытки")
        node = self.nodes[self.pending]
        if "config_key" in result and result["config_key"] != node["config_key"]:
            raise ValueError("Результат относится к другой конфигурации")
        score, status = result.get("score"), result.get("status", "failed")
        if status == "ok" and not self._finite_score(score):
            status = "invalid_metric"
        wall = result.get("wall_seconds", 0.)
        if not self._finite_score(wall) or wall < 0:
            raise ValueError("Некорректная длительность попытки")
        node.update(status=status, score=float(score) if status == "ok" else None,
                    wall_seconds=float(wall), error_type=result.get("error_type"),
                    error=str(result.get("error", result.get("reason", "")))[-1800:] or None,
                    directory=result.get("directory"))
        self.pending = None

    @property
    def best_node(self):
        return deepcopy(max((n for n in self.nodes if n["status"] == "ok"), key=self._rank))

    def to_dict(self):
        return {"policy": self.policy, "seed": self.seed, "max_attempts": self.max_attempts,
                "attempts": len(self.nodes)-1, "explore_probability": self.explore_probability,
                "actions": deepcopy(self.actions), "action_order": list(self.action_order),
                "nodes": deepcopy(self.nodes), "best_node_id": self.best_node["id"],
                "pending_node_id": self.pending, "exhausted": self.exhausted,
                "extra_llm_calls": 0, "extra_llm_tokens": 0,
                "wall_seconds": sum(n["wall_seconds"] or 0. for n in self.nodes[1:])}
