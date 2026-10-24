"""Запуск платформы: python3 solution.py task.md.

Секреты передаются только через OPENAI_API_KEY; исходные данные не изменяются.
"""
from __future__ import annotations
import os
from pathlib import Path
import sys
import time

STARTED = time.monotonic()
CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT))
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "7")


def main(argv=None) -> int:
    import argparse
    from dataclasses import replace
    from datetime import datetime, timezone
    import importlib.metadata
    import json
    import traceback
    import threading
    import numpy as np
    from contracts import CLASSIFICATION_METRICS, ExperimentSpec, METRICS, native
    from task_reader import load_task
    from data_io import encode_target, profile_data
    from validation import make_splits
    from metrics import better, evaluate
    from submission import fallback_prediction, write_submission
    from experiment_store import Journal, atomic_json
    from runtime import Budget, Runner
    from llm_client import LLMPlanner
    from research import Research

    parser = argparse.ArgumentParser(description="Автономный агент табличного ML")
    parser.add_argument("task", nargs="?", default="task.md")
    parser.add_argument("--time-limit", type=float, default=float(os.environ.get("ALFA_TIME_LIMIT_SECONDS", "10800")))
    parser.add_argument("--max-experiments", type=int, default=int(os.environ.get("ALFA_MAX_EXPERIMENTS", "18")))
    parser.add_argument("--llm-calls", type=int, default=int(os.environ.get("ALFA_LLM_CALLS", "6")))
    parser.add_argument("--threads", type=int, default=int(os.environ.get("ALFA_THREADS", "7")))
    parser.add_argument("--memory-mb", type=int, default=24000)
    parser.add_argument("--max-iterations", type=int, default=0, help="Сокращённый локальный прогон; 0 — штатный поиск")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args(argv)
    if not 10 <= args.time_limit <= 10800 or not 1 <= args.max_experiments <= 100:
        parser.error("Допустимы 10..10800 секунд и 1..100 экспериментов")
    if args.max_iterations and not 50 <= args.max_iterations <= 6000:
        parser.error("max-iterations: 0 или 50..6000")
    budget = Budget(STARTED, args.time_limit, args.threads, args.memory_mb)
    root = Path(args.task).resolve().parent
    directory = root / ".alfa_runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f") + f"_{os.getpid()}")
    journal = Journal(directory)
    fallback_written = False
    summary = {"status": "initializing", "time_limit_seconds": budget.seconds,
               "threads": budget.threads, "memory_limit_mb": budget.memory_mb, "artifacts": str(directory)}

    def hard_stop():
        # Ограничивается весь агент, включая профилирование, разбиение и API,
        # а не только fit(). CSV заменяется атомарно и остаётся целым.
        try:
            import psutil
            for child in reversed(psutil.Process().children(recursive=True)):
                try:
                    child.kill()
                except psutil.Error:
                    pass
            summary.update(status="deadline", elapsed_seconds=time.monotonic()-STARTED)
            atomic_json(directory / "summary.json", summary)
            print("Достигнут общий лимит; сохранён последний проверенный ответ.", file=sys.stderr, flush=True)
        finally:
            os._exit(0 if fallback_written else 1)

    watchdog = threading.Timer(max(.1, budget.remaining()-.25), hard_stop)
    watchdog.daemon = True
    watchdog.start()

    def finish(status: str, **extra):
        summary.update(status=status, elapsed_seconds=time.monotonic()-STARTED, **extra)
        atomic_json(directory / "summary.json", summary)
        journal.event("finished", status=status, elapsed_seconds=summary["elapsed_seconds"])
        return 0

    try:
        task, train, test, sample, dictionary = load_task(args.task)
        planner = LLMPlanner(journal, budget, 0 if args.no_llm else args.llm_calls)
        if task.metric == "unknown":
            resolved = planner.resolve_metric(task, Path(task.task_path).read_text(encoding="utf-8-sig"))
            if resolved:
                task.metric = resolved
                problem = "classification" if resolved in CLASSIFICATION_METRICS else "regression"
                if problem != task.problem_type:
                    task.problem_type = problem
                    task.classes = sorted(native(train[task.target].unique().tolist()), key=str) if task.classification else []
                    if task.prediction_kind not in {"probability", "probabilities"}:
                        task.prediction_kind = "label" if task.classification else "value"
        y = encode_target(task, train)
        output = root / "submission.csv"
        fallback = fallback_prediction(task, y, len(test))
        write_submission(task, test, fallback, output)
        fallback_written = True
        journal.event("fallback_saved", rows=len(test), target=task.target)
        summary.update(task=task.to_dict(), output=str(output), fallback_only=True)
        atomic_json(directory / "task.json", task.to_dict())
        if task.metric not in METRICS:
            return finish("fallback_unsupported_metric", reason="Метрика не подтверждена; произвольная формула не выполняется")
        if len(train) < 8 or (task.classification and len(task.classes) < 2):
            return finish("fallback_small_or_constant_task")
        profile, groups, times = profile_data(task, train, test, dictionary)
        splits = make_splits(task, y, groups, times)
        atomic_json(directory / "profile.json", profile)
        split_paths = {}
        for split in splits:
            path = directory / f"split_{split.name}.npz"
            np.savez_compressed(path, train=split.train, valid=split.valid)
            split_paths[split.name] = path
            journal.event("validation", phase=split.name, train_rows=len(split.train),
                          valid_rows=len(split.valid), reason=split.reason)
        versions = {}
        for name in ("numpy", "pandas", "scikit-learn", "catboost", "lightgbm", "openai"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "unavailable"
        summary.update(profile=profile, versions=versions, python=sys.version)
        runner = Runner(directory, task, profile, budget)
        research = Research(task, profile, planner)
        history, successes, executed = [], [], set()
        best_score, best_result, no_improvement = None, None, 0
        search_split = splits[0]

        def predictions(result):
            with np.load(Path(result["directory"]) / "predictions.npz", allow_pickle=False) as data:
                return data["valid"].copy(), data["test"].copy()

        for attempt in range(args.max_experiments):
            if budget.until(.70) < 4:
                break
            config = research.next(history)
            if config is None:
                break
            if args.max_iterations:
                config = replace(config, iterations=min(config.iterations, args.max_iterations))
            if config.key in executed:
                continue
            executed.add(config.key)
            timeout = min(budget.until(.70), max(12, budget.seconds*.12))
            journal.event("experiment_started", phase="search", model=config.model,
                          hypothesis=config.hypothesis, config_key=config.key)
            result = runner.run(config, split_paths["search"], "search", timeout)
            history.append(result)
            journal.event("experiment_result", **result)
            if result["status"] != "ok":
                no_improvement += 1
                continue
            successes.append(result)
            if better(task, result["score"], best_score):
                _, prediction = predictions(result)
                write_submission(task, test, prediction, output)
                best_score, best_result, no_improvement = result["score"], result, 0
                journal.event("best_updated", phase="search", model=config.model, score=best_score)
            else:
                no_improvement += 1
            if no_improvement >= 6 and len(successes) >= 5:
                journal.event("search_stopped", reason="Шесть экспериментов без улучшения")
                break
        if not successes:
            return finish("fallback_no_successful_experiments", experiments=len(history))
        successes.sort(key=lambda r: r["score"], reverse=task.maximize)
        candidates = successes[:2]
        selected = [candidates[0]]
        weights = [1.0]
        # Вес ансамбля выбирается на search и проверяется неизменным на confirm.
        blend_weight = None
        if len(candidates) == 2:
            a, _ = predictions(candidates[0])
            b, _ = predictions(candidates[1])
            blend_score = candidates[0]["score"]
            for weight in (.25, .5, .75):
                value = evaluate(task, y[search_split.valid], weight*a+(1-weight)*b)[task.metric]
                if better(task, value, blend_score, tolerance=.001):
                    blend_score, blend_weight = value, weight
        confirmed = []
        if "confirm" in split_paths:
            for candidate in candidates:
                remaining_candidates = len(candidates)-len(confirmed)
                allowance = budget.until(.84) / max(1, remaining_candidates)
                estimate = candidate["wall_seconds"] * len(splits[1].train)/max(1, candidate["n_train"])
                if allowance < max(4, min(estimate*.7, 15)):
                    break
                config = ExperimentSpec.from_dict(candidate["config"], task)
                journal.event("experiment_started", phase="confirm", model=config.model)
                result = runner.run(config, split_paths["confirm"], "confirm", allowance)
                journal.event("experiment_result", **result)
                if result["status"] == "ok":
                    confirmed.append(result)
            if confirmed:
                winner = sorted(confirmed, key=lambda r: r["score"], reverse=task.maximize)[0]
                selected, weights = [winner], [1.]
                if len(confirmed) == 2 and blend_weight is not None:
                    a, _ = predictions(confirmed[0])
                    b, _ = predictions(confirmed[1])
                    mixed = blend_weight*a+(1-blend_weight)*b
                    score = evaluate(task, y[splits[1].valid], mixed)[task.metric]
                    if better(task, score, winner["score"], tolerance=.001):
                        selected, weights = confirmed, [blend_weight, 1-blend_weight]
                        journal.event("ensemble_confirmed", phase="confirm", score=score, weights=weights)
        # При единственном разбиении выбираем модель без дополнительного подбора ансамбля.
        selected_prediction = sum(w*predictions(r)[1] for w, r in zip(weights, selected))
        write_submission(task, test, selected_prediction, output)
        summary.update(fallback_only=False, selected=[{"config": r["config"], "score": r["score"],
                       "phase": r["phase"], "best_iterations": r["best_iterations"]} for r in selected], weights=weights)
        full_split = directory / "split_refit.npz"
        np.savez_compressed(full_split, train=np.arange(len(train)), valid=np.empty(0, dtype=int))
        refits = []

        def estimate_refit(candidate):
            startup = max(2.0, candidate["wall_seconds"]-candidate["training_seconds"]
                          -candidate["prediction_seconds"])
            return max(8.0, candidate["training_seconds"]*len(train)/max(1,candidate["n_train"])*1.3
                       +candidate["prediction_seconds"]+startup+3)

        for index, candidate in enumerate(selected):
            available = budget.until(.97)
            estimate = estimate_refit(candidate)
            estimates_left = sum(estimate_refit(r) for r in selected[index:])
            if available < estimates_left:
                journal.event("refit_skipped", reason="Недостаточно резерва; сохранён проверенный ответ")
                break
            config = ExperimentSpec.from_dict(candidate["config"], task)
            journal.event("experiment_started", phase="refit", model=config.model)
            result = runner.run(config, full_split, "refit", min(available, estimate*1.5),
                                refit=True, fixed_iterations=candidate["best_iterations"])
            journal.event("experiment_result", **result)
            if result["status"] != "ok":
                break
            refits.append(result)
        if len(refits) == len(selected):
            prediction = sum(w*predictions(r)[1] for w, r in zip(weights, refits))
            write_submission(task, test, prediction, output)
            summary["refit_completed"] = True
        else:
            summary["refit_completed"] = False
        return finish("completed", experiments=len(history), successful_experiments=len(successes),
                      search_best_score=successes[0]["score"], llm_calls=planner.calls, llm_tokens=planner.tokens)
    except Exception as error:
        # API-ошибки перехватываются внутри клиента и не попадают в этот обработчик.
        journal.event("fatal_error", error_type=type(error).__name__, reason=str(error)[-1500:])
        traceback.print_exc()
        summary.update(error_type=type(error).__name__, reason=str(error)[-1500:])
        if fallback_written:
            return finish("recovered_with_submission")
        atomic_json(directory / "summary.json", summary)
        return 1
    finally:
        watchdog.cancel()


if __name__ == "__main__":
    raise SystemExit(main())
