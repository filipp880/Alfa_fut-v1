"""Fixed three-family OOF weights, fitted exclusively inside search-train."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from contracts import parse_experiment
from data_io import profile_data
from experiment_store import atomic_json
from metrics import better, evaluate
from runtime import Runner
from validation import Split


FAMILIES = ("boosting", "forest", "linear")
MODEL_FAMILY = {"catboost": "boosting", "lightgbm": "boosting", "extra_trees": "forest", "linear": "linear"}
GRID_RESERVE_SECONDS = 15.


class OOFError(ValueError):
    pass


def _indices(value, size, empty=False):
    value = np.asarray(value)
    if (value.ndim != 1 or value.dtype.kind not in "iu" or (not empty and not len(value))
            or len(np.unique(value)) != len(value)
            or (len(value) and (value.min() < 0 or value.max() >= size))):
        raise OOFError("invalid_indices")
    return value.astype(np.int64, copy=False)


def _index_hash(value):
    return hashlib.sha256(np.asarray(value, dtype="<i8").tobytes()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_family_candidates(task, successes):
    """Best search result per broad family; no OOF data are consulted."""
    selected = {}
    for result in successes:
        family = MODEL_FAMILY.get(result.get("config", {}).get("model"))
        if family is None or result.get("status") != "ok" or result.get("phase", "search") != "search":
            continue
        try:
            score = result["score"]
            count = result["best_iterations"]
            if isinstance(score, bool) or not math.isfinite(float(score)) or type(count) is not int or count < 1:
                continue
            config = parse_experiment(result["config"], task)
            if config.key != result["config_key"]:
                continue
        except (ValueError, TypeError, KeyError, OverflowError):
            continue
        previous = selected.get(family)
        if previous is None or better(task, float(score), float(previous["score"]), tolerance=0.):
            selected[family] = result
    if set(selected) != set(FAMILIES):
        return []
    # Freeze nested configs independently of the mutable search journal.
    return [json.loads(json.dumps(selected[family], allow_nan=False)) for family in FAMILIES]


def _check_folds(task, y, groups, times, outer, folds):
    covered = []
    for fold in folds:
        tr, va = _indices(fold.train, len(y)), _indices(fold.valid, len(y))
        if (len(tr) < 2 or np.intersect1d(tr, va).size
                or not np.isin(np.r_[tr, va], outer).all()
                or np.intersect1d(groups[tr], groups[va]).size):
            return False
        if times is not None and np.max(times[tr]) >= np.min(times[va]):
            return False
        expected_train = (outer[times[outer] < np.min(times[va])] if times is not None
                          else np.setdiff1d(outer, va))
        if not np.array_equal(np.sort(tr), expected_train):
            return False
        if task.classification:
            axis = np.arange(len(task.classes))
            if not (np.array_equal(np.unique(y[tr]), axis) and np.array_equal(np.unique(y[va]), axis)):
                return False
        covered.extend(va.tolist())
    if len(folds) < 2 or len(covered) != len(set(covered)):
        return False
    uncovered = np.setdiff1d(outer, covered)
    if times is None:
        return not len(uncovered)
    return bool(len(uncovered) and np.max(times[uncovered]) < np.min(times[covered]))


def make_oof_splits(task, y, groups, times, search_split, max_folds=3):
    """Global row indices; temporal folds leave an unscored warmup prefix."""
    try:
        y, groups = np.asarray(y), np.asarray(groups)
        if y.ndim != 1 or groups.ndim != 1 or len(groups) != len(y):
            return []
        outer = np.sort(_indices(search_split.train, len(y)))
        holdout = _indices(search_split.valid, len(y), empty=True)
        if np.intersect1d(outer, holdout).size or len(outer) < 4:
            return []
        local_y, local_groups = y[outer], groups[outer]
        if task.classification:
            if (local_y.dtype.kind not in "iu" or len(task.classes) < 2
                    or not np.array_equal(np.unique(local_y), np.arange(len(task.classes)))):
                return []
        elif not np.isfinite(local_y).all():
            return []
        maximum = min(3, int(max_folds))
        if times is not None:
            times = np.asarray(times)
            if times.ndim != 1 or len(times) != len(y) or not np.isfinite(times[outer]).all():
                return []
            levels = np.unique(times[outer])
            for count in range(min(maximum, len(levels)-1), 1, -1):
                blocks = np.array_split(levels, count+1)
                folds = [Split(f"oof_{i}", outer[times[outer] < block[0]],
                               outer[np.isin(times[outer], block)], "Expanding temporal OOF inside search-train")
                         for i, block in enumerate(blocks[1:])]
                if _check_folds(task, y, groups, times, outer, folds):
                    return folds
            return []
        unique_groups = len(np.unique(local_groups))
        grouped = unique_groups < len(outer)  # Every duplicate matters, including fractions below 1%.
        maximum = min(maximum, unique_groups if grouped else len(outer)//2)
        if task.classification:
            maximum = min(maximum, int(np.bincount(local_y).min()))
            if grouped:
                maximum = min(maximum, min(len(np.unique(local_groups[local_y == c]))
                                           for c in range(len(task.classes))))
        for count in range(maximum, 1, -1):
            if task.classification:
                cv = StratifiedGroupKFold(count, shuffle=True, random_state=42) if grouped else StratifiedKFold(
                    count, shuffle=True, random_state=42)
            else:
                cv = GroupKFold(count) if grouped else KFold(count, shuffle=True, random_state=42)
            parts = cv.split(outer, local_y, local_groups) if grouped else cv.split(outer, local_y)
            folds = [Split(f"oof_{i}", outer[tr], outer[va],
                           "Group OOF inside search-train" if grouped else "OOF inside search-train")
                     for i, (tr, va) in enumerate(parts)]
            if _check_folds(task, y, groups, None, outer, folds):
                return folds
        return []
    except (ValueError, TypeError, IndexError, OverflowError):
        return []


def _prediction(task, value, rows):
    value = np.asarray(value)
    shape = (rows, len(task.classes)) if task.classification else (rows,)
    if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise OOFError("invalid_prediction_shape_or_values")
    value = value.astype(float, copy=False)
    if task.classification and ((value < 0).any() or (value > 1).any()
                               or not np.allclose(value.sum(axis=1), 1., rtol=0., atol=1e-7)):
        raise OOFError("invalid_class_axis_or_probabilities")
    return value


def _score(task, y, prediction):
    """Only the requested metric; avoid computing unrelated metrics on every grid point."""
    if task.classification and task.metric in {"macro_f1", "weighted_f1", "f1", "accuracy", "balanced_accuracy"}:
        k = len(task.classes)
        matrix = np.bincount(y*k + prediction.argmax(axis=1), minlength=k*k).reshape(k, k)
        actual, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
        denominator = actual + predicted
        f1 = np.divide(2*np.diag(matrix), denominator, out=np.zeros(k), where=denominator > 0)
        if task.metric == "macro_f1":
            return float(f1.mean())
        if task.metric == "weighted_f1":
            return float(np.dot(f1, actual)/len(y))
        if task.metric == "f1" and k == 2:
            return float(f1[task.positive_index])
        if task.metric == "accuracy":
            return float(np.trace(matrix)/len(y))
        if task.metric == "balanced_accuracy":
            return float(np.mean(np.diag(matrix)[actual > 0]/actual[actual > 0]))
    return evaluate(task, y, prediction)[task.metric]


def fit_simplex_weights(task, y, predictions, deadline=None):
    """Fit a closed grid: 0.1 simplex, equal thirds, and quarter-weight pairs."""
    y, matrix = np.asarray(y), np.asarray(predictions)
    expected_ndim = 3 if task.classification else 2
    if (y.ndim != 1 or not len(y) or matrix.ndim != expected_ndim
            or matrix.shape[0] != len(y) or matrix.shape[1:2] != (3,)):
        raise OOFError("invalid_oof_matrix")
    if task.classification:
        if y.dtype.kind not in "iu" or not np.isin(y, np.arange(len(task.classes))).all():
            raise OOFError("invalid_oof_target_axis")
    elif not np.isfinite(y).all():
        raise OOFError("invalid_oof_target")
    for member in range(3):
        _prediction(task, matrix[:, member], len(y))
    grid = {(i/10, j/10, (10-i-j)/10) for i in range(11) for j in range(11-i)}
    grid.add((1/3, 1/3, 1/3))
    for a in range(3):
        for b in range(a+1, 3):
            for weight in (.25, .5, .75):
                weights = [0., 0., 0.]
                weights[a], weights[b] = weight, 1-weight
                grid.add(tuple(weights))
    # Stable ties prefer fewer nonzero components, then the frozen family order.
    grid = sorted(grid, key=lambda weights: (sum(w > 0 for w in weights), tuple(-w for w in weights)))
    best_score, best_weights = None, None
    for weights in grid:
        if deadline is not None and time.monotonic() >= deadline:
            raise OOFError("weight_grid_deadline")
        prediction = sum(weight*matrix[:, i] for i, weight in enumerate(weights))
        score = _score(task, y, prediction)
        if deadline is not None and time.monotonic() >= deadline:
            raise OOFError("weight_grid_deadline")
        if better(task, score, best_score, tolerance=1e-12):
            best_score, best_weights = float(score), weights
    if best_weights is None:
        raise OOFError("no_finite_weight_score")
    return {"weights": list(best_weights), "score": best_score, "evaluated": len(grid)}


def _estimate(result, inner_rows):
    wall = float(result["wall_seconds"])
    training = float(result.get("training_seconds", wall*.7))
    prediction = float(result.get("prediction_seconds", 0.))
    rows = int(result["n_train"])
    if rows < 1 or not all(math.isfinite(v) and v >= 0 for v in (wall, training, prediction)):
        raise OOFError("invalid_search_timing")
    startup = max(1., wall-training-prediction)
    # Runner reserves startup/serialization time outside fit(); do not give a tiny
    # model a timeout that leaves no time after that reserve and Python imports.
    return max(8., 1.3*(training+prediction)*inner_rows/rows + startup + 5.)


def propose_oof_plan(task, train, test, dictionary, y, groups, times, search_split,
                     successes, runner, budget, directory, journal):
    """Return only a complete fixed OOF plan. Parent performs the separate search/confirm guards."""
    started = time.monotonic()
    out = Path(directory)/"oof"
    state = {"status": "running", "families": list(FAMILIES), "grid_reserve_seconds": GRID_RESERVE_SECONDS,
             "fit_attempts": 0, "completed_fits": 0, "validated_fits": 0}

    def finish(status, reason=None, **extra):
        state.update(status=status, elapsed_seconds=time.monotonic()-started, **extra)
        if reason is not None:
            state["reason"] = reason
        atomic_json(out/"status.json", state)

    try:
        out.mkdir(parents=True, exist_ok=False)
    except OSError:
        return None
    stage = "setup"
    try:
        seconds = min(float(budget.until(.70)), .12*float(budget.seconds))
        deadline = started + max(0., seconds)
        state["stage_budget_seconds"] = max(0., seconds)
        if seconds <= GRID_RESERVE_SECONDS:
            finish("skipped", "insufficient_budget")
            return None
        candidates = select_family_candidates(task, successes)
        if not candidates:
            finish("skipped", "missing_three_families")
            return None
        y, groups = np.asarray(y), np.asarray(groups)
        if len(y) != len(train):
            raise OOFError("target_rows_mismatch")
        folds = make_oof_splits(task, y, groups, times, search_split)
        if len(folds) < 2:
            finish("skipped", "insufficient_independent_folds")
            return None
        outer = np.sort(_indices(search_split.train, len(y)))
        times = None if times is None else np.asarray(times)
        if not _check_folds(task, y, groups, times, outer, folds):
            raise OOFError("invalid_fold_proof")
        covered = np.sort(np.concatenate([fold.valid for fold in folds]))
        if len(np.unique(covered)) != len(covered):
            raise OOFError("duplicate_oof_coverage")
        axis_width = len(task.classes) if task.classification else 1
        cube_bytes = len(covered)*3*axis_width*8
        transient_rows = max(len(test)+2*max(len(fold.valid) for fold in folds), 3*len(covered))
        peak_bytes = cube_bytes + transient_rows*axis_width*8
        memory_cap = min(256*2**20, .1*float(budget.memory_mb)*2**20)
        state.update(estimated_cube_bytes=cube_bytes, estimated_peak_bytes=peak_bytes, memory_cap_bytes=memory_cap)
        if peak_bytes > memory_cap:
            finish("skipped", "matrix_memory_budget")
            return None
        estimates = [_estimate(candidate, len(fold.train)) for fold in folds for candidate in candidates]
        state.update(estimated_fit_seconds=sum(estimates), search_train_indices_sha256=_index_hash(outer),
                     oof_indices_sha256=_index_hash(covered), oof_rows=len(covered),
                     warmup_rows=len(outer)-len(covered), folds=[], jobs=[])
        if time.monotonic()+sum(estimates)+GRID_RESERVE_SECONDS > deadline:
            finish("skipped", "estimated_fits_exceed_budget")
            return None
        frozen = [{"family": family, "config_key": row["config_key"], "config": row["config"],
                   "best_iterations": row["best_iterations"], "search_score": row["score"]}
                  for family, row in zip(FAMILIES, candidates)]
        atomic_json(out/"configs.json", {"members": frozen, "task_class_axis": list(task.classes)})
        state["configs_sha256"] = _sha(out/"configs.json")
        shape = (len(covered), 3, len(task.classes)) if task.classification else (len(covered), 3)
        matrix = np.full(shape, np.nan)
        job_index = 0
        for number, fold in enumerate(folds):
            fold_dir = out/f"fold{number}"
            fold_dir.mkdir()
            split_path = fold_dir/"split.npz"
            np.savez_compressed(split_path, train=fold.train, valid=fold.valid)
            profile, _, _ = profile_data(task, train.iloc[fold.train], test, dictionary)
            profile["time_column"] = runner.profile.get("time_column")
            profile["profile_scope"] = "fold_train_only"
            atomic_json(fold_dir/"profile.json", profile)
            state["folds"].append({"name": fold.name, "split_sha256": _sha(split_path),
                "train_rows": len(fold.train), "valid_rows": len(fold.valid),
                "train_indices_sha256": _index_hash(fold.train), "valid_indices_sha256": _index_hash(fold.valid),
                "group_disjoint": True, "strictly_past": times is not None,
                "within_search_train": True, "reason": fold.reason})
            fold_runner = Runner(fold_dir, task, profile, budget, worker_module=getattr(runner, "worker_module", "worker"))
            if hasattr(runner, "code_root"):
                fold_runner.code_root = runner.code_root
            for member, candidate in enumerate(candidates):
                if time.monotonic()+sum(estimates[job_index:])+GRID_RESERVE_SECONDS > deadline:
                    finish("discarded" if state["jobs"] else "skipped", "remaining_budget_insufficient")
                    return None
                allowance = min(estimates[job_index]*1.5,
                                deadline-time.monotonic()-GRID_RESERVE_SECONDS-sum(estimates[job_index+1:]))
                stage = "fold_fit"
                config = parse_experiment(candidate["config"], task)
                state["fit_attempts"] += 1
                state["active_fit"] = {"fold": fold.name, "family": FAMILIES[member], "config_key": config.key,
                                       "fixed_iterations": candidate["best_iterations"]}
                finish("running")
                job_started = time.monotonic()
                result = fold_runner.run(config, split_path, fold.name, allowance, refit=True,
                                         fixed_iterations=candidate["best_iterations"])
                if result.get("status") == "ok":
                    state["completed_fits"] += 1
                if (result.get("status") != "ok" or result.get("config_key") != candidate["config_key"]
                        or result.get("best_iterations") != candidate["best_iterations"]
                        or result.get("n_train") != len(fold.train) or result.get("training_stopped_for_budget")
                        or result.get("evaluation_used_for_fit")):
                    raise OOFError("incomplete_or_changed_fold_fit")
                prediction_path = Path(result["directory"])/"predictions.npz"
                if prediction_path.is_symlink() or not prediction_path.resolve().is_relative_to(fold_dir.resolve()):
                    raise OOFError("foreign_prediction_path")
                with np.load(prediction_path, allow_pickle=False) as values:
                    if set(values.files) != {"valid", "test", "valid_indices"}:
                        raise OOFError("unexpected_prediction_fields")
                    actual_indices = _indices(values["valid_indices"], len(y))
                    if not np.array_equal(actual_indices, fold.valid):
                        raise OOFError("changed_validation_indices")
                    valid = _prediction(task, values["valid"], len(fold.valid)).copy()
                    _prediction(task, values["test"], len(test))
                independent_score = _score(task, y[fold.valid], valid)
                if not np.isclose(independent_score, result.get("score", np.nan), rtol=1e-10, atol=1e-10):
                    raise OOFError("fold_score_mismatch")
                matrix[np.searchsorted(covered, fold.valid), member] = valid
                state["jobs"].append({"fold": fold.name, "family": FAMILIES[member],
                    "model": config.model, "config_key": config.key, "fixed_iterations": candidate["best_iterations"],
                    "actual_iterations": result["best_iterations"], "prediction_sha256": _sha(prediction_path),
                    "directory": str(prediction_path.parent.relative_to(out)), "score": float(independent_score),
                    "wall_seconds": float(result.get("wall_seconds", time.monotonic()-job_started)),
                    "training_seconds": float(result.get("training_seconds", 0.)),
                    "prediction_seconds": float(result.get("prediction_seconds", 0.))})
                state["validated_fits"] += 1
                state.pop("active_fit", None)
                job_index += 1
        stage = "weight_grid"
        if not np.isfinite(matrix).all() or time.monotonic()+GRID_RESERVE_SECONDS > deadline:
            raise OOFError("matrix_or_grid_budget_incomplete")
        np.savez_compressed(out/"raw_matrix.npz", indices=covered, predictions=matrix,
                            target=y[covered], member_keys=np.asarray([row["config_key"] for row in candidates]),
                            warmup_indices=np.setdiff1d(outer, covered),
                            class_indices=np.arange(len(task.classes), dtype=np.int64))
        fitted = fit_simplex_weights(task, y[covered], matrix, deadline=deadline)
        if time.monotonic() > deadline:
            raise OOFError("weight_grid_deadline")
        weights = fitted["weights"]
        selected = [i for i, weight in enumerate(weights) if weight > 0]
        plan = {"members": [candidates[i]["config_key"] for i in selected],
                "weights": [weights[i] for i in selected], "oof_score": fitted["score"]}
        finish("completed", weights=weights, selected_plan=plan, oof_score=fitted["score"],
               grid_points=fitted["evaluated"], raw_matrix_sha256=_sha(out/"raw_matrix.npz"))
        return plan
    except Exception as error:
        try:
            finish("discarded", str(error) if isinstance(error, OOFError) else "oof_failed",
                   failed_stage=stage, error_type=type(error).__name__)
        except OSError:
            pass
        return None
