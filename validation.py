"""Фиксированные поисковое и подтверждающее разбиения."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold, GroupKFold

from contracts import TaskSpec


@dataclass
class Split:
    name: str
    train: np.ndarray
    valid: np.ndarray
    reason: str

    def check(self, n: int, y: np.ndarray, classification: bool):
        if not len(self.train) or not len(self.valid):
            raise ValueError("Пустая часть разбиения")
        if np.intersect1d(self.train, self.valid).size:
            raise ValueError("Пересечение обучения и валидации")
        if min(self.train.min(), self.valid.min()) < 0 or max(self.train.max(), self.valid.max()) >= n:
            raise ValueError("Индекс разбиения вне таблицы")
        if classification and set(np.unique(y[self.valid])) - set(np.unique(y[self.train])):
            raise ValueError("Класс валидации отсутствует в обучении")
        return self


def make_splits(task: TaskSpec, y: np.ndarray, groups: np.ndarray,
                time_values: np.ndarray | None, seed: int = 42) -> list[Split]:
    n = len(y)
    indices = np.arange(n)
    if n < 8:
        raise ValueError("Для надёжной валидации нужно не менее 8 строк")
    if time_values is not None:
        levels = np.unique(time_values)
        if len(levels) > 20:
            # Для уникальных дат/времени один последний timestamp не является
            # достаточной валидацией: резервируем последовательные блоки.
            search_cut = levels[int(len(levels)*.6)]
            confirm_cut = levels[int(len(levels)*.8)]
            result = [Split("search", indices[time_values < search_cut],
                            indices[(time_values >= search_cut) & (time_values < confirm_cut)],
                            "Хронологические блоки: 60% обучение, 20% поиск, 20% подтверждение"),
                      Split("confirm", indices[time_values < confirm_cut], indices[time_values >= confirm_cut],
                            "Отдельный последний хронологический блок")]
            return [s.check(n, y, task.classification) for s in result]
        cuts = [levels[-2], levels[-1]] if len(levels) >= 4 else [levels[-1]]
        result = [Split("search" if i == 0 else "confirm",
                        indices[time_values < cut], indices[time_values == cut],
                        "Будущий тест: обучение строго раньше проверяемого периода")
                  for i, cut in enumerate(cuts)]
        return [s.check(n, y, task.classification) for s in result]
    duplicate_fraction = 1 - len(np.unique(groups)) / n
    if task.classification:
        counts = np.bincount(y)
        minimum = int(counts[counts > 0].min())
        if minimum < 2:
            # Одиночные классы остаются только в обучении. Все классы учитываются метрикой.
            eligible = indices[counts[y] >= 2]
            singleton = indices[counts[y] < 2]
            if len(eligible) < 4:
                raise ValueError("Слишком мало повторных наблюдений классов")
            cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            tr, va = next(cv.split(eligible, y[eligible]))
            return [Split("search", np.concatenate([eligible[tr], singleton]), eligible[va],
                          "Редкие одиночные классы оставлены в обучении").check(n, y, True)]
        folds = min(5, minimum)
        if duplicate_fraction >= .01:
            independent = [len(np.unique(groups[y == c])) for c in np.unique(y)]
            folds = min(folds, min(independent))
            if folds < 2:
                raise ValueError("Невозможно разделить классы без пересечения групп")
            cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
            parts = [va for _, va in cv.split(indices, y, groups)]
            reason = "Стратификация с непересекающимися группами одинаковых признаков"
        else:
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            parts = [va for _, va in cv.split(indices, y)]
            reason = "Стратификация по классам"
    else:
        folds = min(5, n // 3)
        if duplicate_fraction >= .01 and len(np.unique(groups)) >= folds:
            cv = GroupKFold(n_splits=folds)
            parts = [va for _, va in cv.split(indices, y, groups)]
            reason = "Непересекающиеся группы одинаковых признаков"
        else:
            cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
            parts = [va for _, va in cv.split(indices)]
            reason = "Фиксированное случайное разбиение"
    if len(parts) < 3:
        valid = parts[0]
        return [Split("search", np.setdiff1d(indices, valid), valid, reason).check(n, y, task.classification)]
    # Отложенная confirm-часть не участвует ни в обучении, ни в оценке поисковых моделей.
    search_valid, confirm_valid = parts[0], parts[1]
    search_train = np.setdiff1d(indices, np.concatenate([search_valid, confirm_valid]))
    result = [Split("search", search_train, search_valid, reason),
              Split("confirm", np.setdiff1d(indices, confirm_valid), confirm_valid, reason)]
    for split in result:
        split.check(n, y, task.classification)
        if duplicate_fraction >= .01 and "групп" in reason:
            if np.intersect1d(groups[split.train], groups[split.valid]).size:
                raise ValueError("Пересечение групп")
    return result
