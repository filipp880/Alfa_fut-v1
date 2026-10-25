"""Повтор фиксированных планов Kaggle на прежнем confirm с другим seed.

Запускать после завершения основного исследования в отдельном network=none
контейнере: python3 /helper/repeat_kaggle.py /task --seed 43.
Импортируется именно snapshot модулей из case, основной submission не меняется.
Это проверка шума обучения на уже изученном confirm, а не новый holdout.
"""
from pathlib import Path
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
import sys
import threading
import time
import traceback

STARTED = time.monotonic()
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--time-limit", type=int, default=900)
    parser.add_argument("--trial-seconds", type=int, default=420)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**31 or not 30 <= args.time_limit <= 900 or not 10 <= args.trial_seconds <= 900:
        parser.error("Недопустимый seed или бюджет повтора")
    work, save_callback = None, None

    def deadline():
        try:
            import psutil
            for child in reversed(psutil.Process().children(recursive=True)):
                try:
                    child.kill()
                except psutil.Error:
                    pass
            if save_callback is not None:
                save_callback("deadline")
            elif work is not None:
                (work/"deadline.json").write_text(json.dumps({"status": "deadline", "elapsed_seconds": time.monotonic()-STARTED}), encoding="utf-8")
        finally:
            os._exit(124)

    # В лимит входят также чтение CSV, импорты и проверка исходных артефактов.
    timer = threading.Timer(max(.1, args.time_limit-(time.monotonic()-STARTED)), deadline)
    timer.daemon = True
    timer.start()
    case = args.case.resolve(strict=True)
    snapshots = sorted(case.glob(".kaggle_research/*/frozen_before_confirmation.json"))
    if not snapshots:
        parser.error("В case отсутствует замороженное исследование")
    source = snapshots[-1].parent
    original = json.loads((source/"summary.json").read_text(encoding="utf-8"))
    frozen = json.loads(snapshots[-1].read_text(encoding="utf-8"))
    if original.get("status") != "completed":
        parser.error("Основное исследование должно завершиться до отдельного повтора")
    sys.path.insert(0, str(case))
    # Проверка происхождения важнее версии локального root, который мог измениться.
    modules = {name: importlib.import_module(name) for name in
               ("contracts", "data_io", "experiment_store", "kaggle_specs", "metrics", "runtime", "task_reader")}
    if any(Path(module.__file__).resolve().parent != case for module in modules.values()):
        raise RuntimeError("Runtime-модули должны импортироваться из snapshot case")
    import numpy as np
    encode_target = modules["data_io"].encode_target
    atomic_json = modules["experiment_store"].atomic_json
    parse_research_spec = modules["kaggle_specs"].parse_research_spec
    evaluate = modules["metrics"].evaluate
    Budget, Runner = modules["runtime"].Budget, modules["runtime"].Runner
    load_task = modules["task_reader"].load_task
    task, train, _, _, _ = load_task(case/"task.md")
    if task.target != original["task"]["target"] or task.metric != original["task"]["metric"]:
        raise ValueError("Контракт задачи отличается от основного исследования")
    y = encode_target(task, train)
    split_path = source/"confirm.npz"
    with np.load(split_path, allow_pickle=False) as values:
        valid_indices = values["valid"].copy()
        train_indices = values["train"].copy()
    if not len(valid_indices) or not len(train_indices) or np.intersect1d(valid_indices, train_indices).size:
        raise ValueError("Некорректное исходное confirm-разбиение")
    if any(i < 0 or i >= len(train) for i in np.r_[valid_indices, train_indices]):
        raise ValueError("Индексы confirm выходят за размер исходного train")
    # Сравниваем исходный и новый планы даже при откате основного агента к baseline.
    # Иначе selected == baseline превратил бы проверку устойчивости в ET против ET.
    plans = {"baseline": original["baseline"], "integrated": original["integrated"]}
    candidates = {record["config_key"]: record for record in frozen["candidates"]}
    original_predictions = {}
    for label, plan in plans.items():
        registered_plans = frozen["baseline_plans"] if label == "baseline" else frozen["integrated_plans"]
        if not any(plan["members"] == p["members"] and plan["weights"] == p["weights"] for p in registered_plans):
            raise ValueError("План отсутствует в замороженном наборе")
        if len(plan["members"]) != len(plan["weights"]) or not plan["members"] or any(
                key not in candidates for key in plan["members"]):
            raise ValueError("Некорректные члены плана")
        if any(not isinstance(w, (float, int)) or not math.isfinite(w) or w < 0 for w in plan["weights"]) or not math.isclose(sum(plan["weights"]), 1.):
            raise ValueError("Некорректные замороженные веса")
        filename = plan["predictions_path"]
        if Path(filename).name != filename:
            raise ValueError("Ожидается локальный файл предсказаний плана")
        with np.load(source/filename, allow_pickle=False) as values:
            if not np.array_equal(valid_indices, values["valid_indices"]):
                raise ValueError("Предсказания исходного плана относятся к другому confirm")
            original_predictions[label] = values["valid"].copy()
        score = evaluate(task, y[valid_indices], original_predictions[label])[task.metric]
        if not math.isclose(score, plan["confirmation_score"], rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError("Метрика исходных предсказаний не воспроизводится")
    work = case/".kaggle_repeat"/datetime.now(timezone.utc).strftime(f"%Y%m%dT%H%M%S_%f_seed{args.seed}")
    work.mkdir(parents=True, exist_ok=False)
    budget = Budget(STARTED, args.time_limit, 7, 24000)
    summary = {"status": "running", "seed": args.seed, "metric": task.metric,
               "source_directory": str(source), "time_limit_seconds": args.time_limit,
               "threads": budget.threads, "memory_budget_mb": budget.memory_mb,
               "frozen_sha256": hashlib.sha256(snapshots[-1].read_bytes()).hexdigest(),
               "confirm_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
               "llm_calls": 0, "llm_tokens": 0, "experiments": [], "plans": {},
               "validation_note": "Повтор шума обучения на прежнем confirm; не новый слепой holdout. Конфигурации, итерации и веса фиксированы, retune отсутствует.",
               "snapshot_modules": {name: str(module.__file__) for name, module in modules.items()}}

    def save(status=None):
        if status:
            summary["status"] = status
        summary["elapsed_seconds"] = time.monotonic()-STARTED
        summary["fit_seconds"] = sum(r.get("wall_seconds", 0.) for r in summary["experiments"])
        summary["peak_worker_rss_mb"] = max((r.get("peak_memory_mb", 0.) for r in summary["experiments"]), default=0.)
        atomic_json(work/"summary.json", summary)

    save_callback = save
    save()
    print("Отдельный повтор фиксированных планов: "+str(work), flush=True)
    try:
        runner = Runner(work, task, original["profile"], budget, worker_module="kaggle_worker")
        repeated = {}
        keys = list(dict.fromkeys(key for plan in plans.values() for key in plan["members"]))
        for key in keys:
            record = candidates[key]
            config_data = copy.deepcopy(record["config"])
            config_data["seed"] = args.seed
            if config_data.get("model") == "augmentation":
                config_data["base_config"]["seed"] = args.seed
            config = parse_research_spec(config_data, task)
            iterations = record["best_iterations"]
            if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
                raise ValueError("Отсутствует фиксированное число итераций исходного кандидата")
            result = runner.run(config, split_path, "repeat_confirm", min(args.trial_seconds, budget.remaining()),
                                refit=True, fixed_iterations=iterations)
            result["source_config_key"] = key
            result["fixed_iterations"] = iterations
            summary["experiments"].append(result)
            repeated[key] = result
            print(f'{key}: {result["status"]}; {task.metric}={result.get("score")}', flush=True)
            save()
        for label, plan in plans.items():
            results = [repeated[key] for key in plan["members"]]
            details = {"members": plan["members"], "weights": plan["weights"],
                       "original_score": plan["confirmation_score"],
                       "fit_seconds": sum(r.get("wall_seconds", 0.) for r in results),
                       "peak_worker_rss_mb": max((r.get("peak_memory_mb", 0.) for r in results), default=0.)}
            if any(r.get("status") != "ok" for r in results):
                details["status"] = "incomplete"
                summary["plans"][label] = details
                continue
            prediction = None
            for weight, result in zip(plan["weights"], results):
                with np.load(Path(result["directory"])/"predictions.npz", allow_pickle=False) as values:
                    if not np.array_equal(values["valid_indices"], valid_indices):
                        raise ValueError("Runner изменил разбиение confirm")
                    part = weight*values["valid"]
                    prediction = part if prediction is None else prediction+part
            score = evaluate(task, y[valid_indices], prediction)[task.metric]
            filename = label+"_paired_predictions.npz"
            np.savez_compressed(work/filename, valid_indices=valid_indices, y=y[valid_indices],
                                original=original_predictions[label], repeat=prediction)
            details.update(status="ok", repeat_score=score, predictions_path=filename,
                           seed_change=score-plan["confirmation_score"])
            summary["plans"][label] = details
        if all(p.get("status") == "ok" for p in summary["plans"].values()):
            direction = 1. if task.maximize else -1.
            base, integrated = (summary["plans"][name] for name in ("baseline", "integrated"))
            summary["integrated_gain_original"] = direction*(integrated["original_score"]-base["original_score"])
            summary["integrated_gain_repeat"] = direction*(integrated["repeat_score"]-base["repeat_score"])
            summary["integrated_gain_repeat_relative"] = summary["integrated_gain_repeat"]/max(abs(base["repeat_score"]), 1e-12)
            save("completed")
            print(json.dumps(summary["plans"], ensure_ascii=False, indent=2), flush=True)
            return 0
        save("incomplete")
        return 1
    except Exception as error:
        summary["error"] = str(error)[-1800:]
        traceback.print_exc()
        save("failed")
        return 1
    finally:
        timer.cancel()


if __name__ == "__main__":
    raise SystemExit(main())
