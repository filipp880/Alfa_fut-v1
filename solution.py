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
    from contracts import CLASSIFICATION_METRICS, ExperimentSpec, METRICS, native, parse_experiment
    from task_reader import load_task
    from data_io import encode_target, profile_data, feature_groups, detect_time
    from validation import make_splits
    from metrics import better, evaluate
    from submission import fallback_prediction, write_submission
    from experiment_store import Journal, atomic_json
    from runtime import Budget, Runner
    from llm_client import LLMPlanner
    from research import Research, search_plan, control_plan, cap_iterations, choose_confirmed_plan
    from submission_audit import write_run_manifest

    parser = argparse.ArgumentParser(description="Автономный агент табличного ML")
    parser.add_argument("task", nargs="?", default="task.md")
    parser.add_argument("--time-limit", type=float, default=float(os.environ.get("ALFA_TIME_LIMIT_SECONDS", "10800")))
    parser.add_argument("--max-experiments", type=int, default=int(os.environ.get("ALFA_MAX_EXPERIMENTS", "18")))
    parser.add_argument("--llm-calls", type=int, default=int(os.environ.get("ALFA_LLM_CALLS", "6")))
    parser.add_argument("--threads", type=int, default=int(os.environ.get("ALFA_THREADS", "7")))
    parser.add_argument("--memory-mb", type=int, default=24000)
    parser.add_argument("--max-iterations", type=int, default=0, help="Сокращённый локальный прогон; 0 — штатный поиск")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-oof", action="store_true", help="Отключить OOF для сравнительного прогона")
    parser.add_argument("--planner", choices=("normal", "sgr"), default=os.environ.get("ALFA_PLANNER_MODE", "normal"))
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
    planner = None

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
            if planner is not None:
                summary.update(llm_calls=planner.calls, llm_tokens=planner.tokens)
            atomic_json(directory / "summary.json", summary)
            print("Достигнут общий лимит; сохранён последний проверенный ответ.", file=sys.stderr, flush=True)
        finally:
            os._exit(0 if fallback_written else 1)

    watchdog = threading.Timer(max(.1, budget.remaining()-.25), hard_stop)
    watchdog.daemon = True
    watchdog.start()

    def finish(status: str, **extra):
        summary.update(status=status, elapsed_seconds=time.monotonic()-STARTED, **extra)
        if planner is not None:
            summary.update(llm_calls=planner.calls, llm_tokens=planner.tokens)
        audit = {}
        if fallback_written:
            manifest = write_run_manifest(directory, task, summary)
            audit = summary["submission_audit"] = manifest["submission"]
            if manifest["status"] != "ok":
                summary["status"] = "failed_submission_audit"
        summary["elapsed_seconds"] = time.monotonic()-STARTED
        atomic_json(directory / "summary.json", summary)
        journal.event("finished", status=summary["status"], elapsed_seconds=summary["elapsed_seconds"],
                      prediction_source=summary.get("prediction_source"),
                      refit_completed=summary.get("refit_completed", False),
                      fallback_only=summary.get("fallback_only", True),
                      llm_calls=summary.get("llm_calls", 0), llm_tokens=summary.get("llm_tokens", 0),
                      submission_rows=audit.get("rows"), submission_sha256=audit.get("sha256"))
        return int(summary["status"] == "failed_submission_audit")

    try:
        task, train, test, sample, dictionary = load_task(args.task)
        planner_class = LLMPlanner
        if args.planner == "sgr":
            from sgr_planner import SGRPlanner
            planner_class = SGRPlanner
        planner = planner_class(journal, budget, 0 if args.no_llm else args.llm_calls)
        summary["planner_mode"] = args.planner
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
        summary.update(task=task.to_dict(), output=str(output), fallback_only=True,
                       prediction_source="fallback")
        atomic_json(directory / "task.json", task.to_dict())
        if task.metric not in METRICS:
            return finish("fallback_unsupported_metric", reason="Метрика не подтверждена; произвольная формула не выполняется")
        if len(train) < 8 or (task.classification and len(task.classes) < 2):
            return finish("fallback_small_or_constant_task")
        groups = feature_groups(task, train)
        time_column, times = detect_time(task, train, test, dictionary)
        splits = make_splits(task, y, groups, times)
        profile, _, _ = profile_data(task, train.iloc[splits[0].train], test, dictionary)
        profile["time_column"] = time_column
        profile["profile_scope"] = "search_train_only"
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
        from history_router import suggest_router_schema, history_schema_diagnostics
        schema = suggest_router_schema(dictionary, task, profile, train.iloc[splits[0].train])
        gate = {"eligible": bool(schema), "schema": schema}
        if schema:
            gate["diagnostics"] = history_schema_diagnostics(schema, train.iloc[splits[0].train][task.feature_columns])
        atomic_json(directory / "operator_gate.json", gate)
        summary["history_operator_eligible"] = bool(schema)
        research = Research(task, profile, planner, include_alternatives=True, history_schema=schema,
                            operator_options=("robust_fallback",))
        history, successes, executed = [], [], set()
        best_score, best_result, no_improvement = None, None, 0
        search_split = splits[0]

        def predictions(result):
            with np.load(Path(result["directory"]) / "predictions.npz", allow_pickle=False) as data:
                phase = "refit" if result["phase"] == "refit" else result["phase"]
                path = directory / "split_refit.npz" if phase == "refit" else split_paths[phase]
                with np.load(path, allow_pickle=False) as expected:
                    valid_indices = expected["valid"]
                np.testing.assert_array_equal(data["valid_indices"], valid_indices)
                valid, predicted = data["valid"].copy(), data["test"].copy()
            if len(valid) != len(valid_indices) or len(predicted) != len(test) or not np.isfinite(valid).all() or not np.isfinite(predicted).all():
                raise ValueError("Неверные размерности или конечность предсказаний worker")
            if len(valid_indices):
                score = evaluate(task, y[valid_indices], valid)[task.metric]
                if not np.isclose(score, result["score"], atol=1e-10, rtol=1e-10):
                    raise ValueError("Метрика worker не воспроизводится в родительском процессе")
            return valid, predicted

        for attempt in range(args.max_experiments):
            if budget.until(.70) < 4:
                break
            config = research.next(history)
            if config is None:
                break
            config = cap_iterations(config, args.max_iterations)
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
            predictions(result)
            if better(task, result["score"], best_score):
                _, prediction = predictions(result)
                write_submission(task, test, prediction, output)
                summary.update(fallback_only=False, prediction_source="search")
                best_score, best_result, no_improvement = result["score"], result, 0
                journal.event("best_updated", phase="search", model=config.model, score=best_score)
            else:
                no_improvement += 1
            if no_improvement >= 6 and len(successes) >= 5 and not getattr(research, "essential_pending", False):
                journal.event("search_stopped", reason="Шесть экспериментов без улучшения")
                break
        if not successes:
            return finish("fallback_no_successful_experiments", experiments=len(history))
        successes.sort(key=lambda r: r["score"], reverse=task.maximize)
        by_key = {r["config_key"]: r for r in successes}
        load_valid = lambda r: predictions(r)[0]
        planned = search_plan(task, successes, y[search_split.valid], load_valid)
        controls = [r for r in successes if r["config"]["model"] not in {"history_router", "proposal"}]
        baseline = control_plan(task, profile, controls or successes, y[search_split.valid], load_valid)
        if not better(task, planned["score"], baseline["score"], .001):
            planned = baseline
        legacy_planned, legacy_baseline = planned, baseline
        summary["oof_selected"] = False
        if args.no_oof:
            summary["oof_status"] = "disabled"
        else:
            try:
                from oof_ensemble import propose_oof_plan
                proposal = propose_oof_plan(task, train, test, dictionary, y, groups, times,
                                           search_split, successes, runner, budget, directory, journal)
                oof_status_path = directory/"oof"/"status.json"
                if oof_status_path.exists():
                    oof_info = json.loads(oof_status_path.read_text(encoding="utf-8"))
                    summary["oof_status"] = oof_info.get("status", "unknown")
                    for field in ("fit_attempts", "completed_fits", "validated_fits", "elapsed_seconds"):
                        summary["oof_"+field] = oof_info.get(field, 0)
                if proposal:
                    keys, weights = proposal["members"], proposal["weights"]
                    if (not 1 <= len(keys) <= 3 or len(set(keys)) != len(keys)
                            or len(weights) != len(keys) or any(key not in by_key for key in keys)
                            or any(isinstance(w,bool) for w in weights)
                            or not np.isfinite(weights).all() or np.min(weights) < 0
                            or not np.isclose(sum(weights),1.,rtol=0,atol=1e-10)):
                        raise ValueError("Недопустимый OOF-план")
                    family = {"catboost":"boosting", "lightgbm":"boosting",
                              "extra_trees":"forest", "linear":"linear"}
                    kinds = [family[by_key[key]["config"]["model"]] for key in keys]
                    if len(set(kinds)) != len(kinds):
                        raise ValueError("Повторяющееся семейство OOF-плана")
                    mixed = sum(weight*predictions(by_key[key])[0]
                                for key,weight in zip(keys,weights))
                    proposed_score = evaluate(task,y[search_split.valid],mixed)[task.metric]
                    if better(task,proposed_score,planned["score"],.001):
                        planned = {"members":keys, "weights":weights, "score":proposed_score}
                        summary["oof_selected"] = True
                    journal.event("oof_plan_evaluated", score=proposed_score,
                                  status="selected" if summary["oof_selected"] else "rejected",
                                  reason="Фиксированные OOF-веса проверены на отдельной search-valid")
            except Exception as error:
                summary.update(oof_status="failed",oof_selected=False)
                planned = legacy_planned
                journal.event("oof_plan_discarded",error_type=type(error).__name__,
                              reason="OOF не заменяет сохранённый рабочий план при ошибке")
        # Сначала подтверждаем старый план и его контроль. Если время кончится,
        # более дорогая OOF-альтернатива не лишит нас прежнего результата.
        candidate_keys = list(dict.fromkeys(legacy_planned["members"] + baseline["members"] + planned["members"]))
        candidates = [by_key[key] for key in candidate_keys]
        summary["oof_selected_on_search"] = summary["oof_selected"]
        atomic_json(directory/"finalists.json", {
            "candidates": [{"config_key": r["config_key"], "config": r["config"],
                            "best_iterations": r["best_iterations"]} for r in candidates],
            "selected_plan": planned, "baseline_plan": baseline,
            "legacy_selected_plan": legacy_planned, "legacy_baseline_plan": legacy_baseline,
            "oof_selected": summary["oof_selected"],
            "selection_rule": "Все планы выбраны до confirm; сначала прежний fallback, затем OOF против его результата",
            "confirmation_early_stopping": False})
        chosen_plan = planned
        selected = [by_key[key] for key in chosen_plan["members"]]
        weights = chosen_plan["weights"]
        confirmed = []
        if "confirm" in split_paths:
            for index, candidate in enumerate(candidates):
                remaining_candidates = len(candidates)-index
                allowance = budget.until(.84) / max(1, remaining_candidates)
                estimate = candidate["wall_seconds"] * len(splits[1].train)/max(1, candidate["n_train"])
                if allowance < max(4, min(estimate*.7, 15)):
                    break
                config = parse_experiment(candidate["config"], task)
                journal.event("experiment_started", phase="confirm", model=config.model)
                result = runner.run(config, split_paths["confirm"], "confirm", allowance,
                                    refit=True, fixed_iterations=candidate["best_iterations"])
                journal.event("experiment_result", **result)
                if result["status"] == "ok":
                    predictions(result)
                    confirmed.append(result)
            if confirmed:
                confirmed_by_key = {r["config_key"]: r for r in confirmed}
                def confirmed_score(plan):
                    if any(key not in confirmed_by_key for key in plan["members"]):
                        return None
                    p = sum(w*predictions(confirmed_by_key[key])[0] for w, key in zip(plan["weights"], plan["members"]))
                    return evaluate(task, y[splits[1].valid], p)[task.metric]
                chosen_plan, confirmation = choose_confirmed_plan(
                    task, planned, legacy_planned, legacy_baseline, confirmed_score,
                    summary["oof_selected"])
                if confirmed_score(chosen_plan) is not None:
                    selected = [confirmed_by_key[key] for key in chosen_plan["members"]]
                    summary["confirmation_score"] = confirmed_score(chosen_plan)
                else:
                    selected = [by_key[key] for key in chosen_plan["members"]]
                weights = chosen_plan["weights"]
                journal.event("frozen_plan_confirmed", **confirmation, selected_plan=chosen_plan)
            elif summary["oof_selected"]:
                chosen_plan = legacy_planned
                selected = [by_key[key] for key in chosen_plan["members"]]
                weights = chosen_plan["weights"]
                journal.event("oof_plan_discarded", reason="Нет завершённого confirm для безопасного сравнения")
        summary["oof_selected"] = bool(summary["oof_selected_on_search"] and chosen_plan == planned)
        # Веса не подбираются заново после confirm; при его отсутствии
        # отдельной проверкой OOF-весов служит search-valid.
        selected_prediction = sum(w*predictions(r)[1] for w, r in zip(weights, selected))
        write_submission(task, test, selected_prediction, output)
        summary.update(fallback_only=False, selected=[{"config": r["config"], "score": r["score"],
                       "phase": r["phase"], "best_iterations": r["best_iterations"]} for r in selected], weights=weights,
                       selected_plan=chosen_plan, prediction_source=selected[0]["phase"])
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
            config = parse_experiment(candidate["config"], task)
            journal.event("experiment_started", phase="refit", model=config.model)
            result = runner.run(config, full_split, "refit", min(available, estimate*1.5),
                                refit=True, fixed_iterations=candidate["best_iterations"])
            journal.event("experiment_result", **result)
            if result["status"] != "ok":
                break
            predictions(result)
            refits.append(result)
        if len(refits) == len(selected):
            prediction = sum(w*predictions(r)[1] for w, r in zip(weights, refits))
            write_submission(task, test, prediction, output)
            summary["refit_completed"] = True
            summary["prediction_source"] = "refit"
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
