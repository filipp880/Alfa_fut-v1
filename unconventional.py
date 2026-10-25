"""Предварительно зафиксированное сравнение трёх альтернатив с текущим агентом."""
from pathlib import Path
import os
import sys
import time

STARTED = time.monotonic()
sys.path.insert(0, str(Path(__file__).resolve().parent))
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"


def main(argv=None):
    import argparse
    from dataclasses import replace
    from datetime import datetime, timezone
    import importlib.metadata
    import hashlib
    import json
    import re
    import threading
    import traceback
    import numpy as np
    import psutil
    from contracts import ExperimentSpec
    from data_io import encode_target, profile_data
    from experiment_store import Journal, atomic_json
    from metrics import better, evaluate
    from novel_models import NovelSpec, portfolio
    from research import Research
    from runtime import Budget, Runner
    from submission import fallback_prediction, write_submission
    from task_reader import load_task
    from validation import make_splits

    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", default="task.md")
    parser.add_argument("--time-limit", type=float, default=10800)
    parser.add_argument("--trial-seconds", type=float, default=360)
    parser.add_argument("--control-trials", type=int, default=8)
    args = parser.parse_args(argv)
    if not 30 <= args.time_limit <= 10800 or not 1 <= args.control_trials <= 8:
        parser.error("Лимит задачи 30..10800 секунд; контроль 1..8 конфигураций")
    root = Path(args.task).resolve().parent
    work = root/".unconventional"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f"))
    journal = Journal(work)
    budget = Budget(STARTED, args.time_limit, 7, 24000)
    summary = {"status": "initializing", "methods": {}, "time_limit_seconds": budget.seconds,
               "validation_policy": "Поиск и подтверждение разделены; без early stopping на подтверждении",
               "network": "none", "experiments": []}
    fallback_written = False

    def save(status=None):
        if status:
            summary["status"] = status
        summary["elapsed_seconds"] = time.monotonic()-STARTED
        atomic_json(work/"summary.json", summary)

    def stop():
        try:
            for child in reversed(psutil.Process().children(recursive=True)):
                try:
                    child.kill()
                except psutil.Error:
                    pass
            save("deadline")
        finally:
            os._exit(0 if fallback_written else 1)

    timer = threading.Timer(max(.1, budget.remaining()-.25), stop)
    timer.daemon = True
    timer.start()
    try:
        task, train, test, _, dictionary = load_task(args.task)
        requirements_path = root/"requirements.txt"
        installed, unused, mismatches = {}, {}, []
        for line in requirements_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            name, expected = re.sub(r"\[[^\]]*\]", "", line.strip()).split("==")
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                unused[name] = expected
                continue
            installed[name] = actual
            if actual != expected:
                mismatches.append({"name": name, "expected": expected, "actual": actual})
        atomic_json(root/"dependency_audit.json", {"installed_approved": installed,
                    "unused_not_installed": unused, "mismatches": mismatches,
                    "requirements_sha256": hashlib.sha256(requirements_path.read_bytes()).hexdigest()})
        if mismatches:
            raise RuntimeError("Версии установленных зависимостей не соответствуют ТЗ")
        y = encode_target(task, train)
        write_submission(task, test, fallback_prediction(task, y, len(test)), root/"submission.csv")
        fallback_written = True
        full_profile, groups, times = profile_data(task, train, test, dictionary)
        splits = make_splits(task, y, groups, times)
        if len(splits) < 2:
            raise ValueError("Для исследования нужны отдельные поиск и подтверждение")
        search, confirm = splits[:2]
        profile, _, _ = profile_data(task, train.iloc[search.train], test, dictionary)
        profile["time_column"] = full_profile["time_column"]
        summary.update(task=task.to_dict(), profile=profile,
                       splits={s.name: {"train": len(s.train), "valid": len(s.valid), "reason": s.reason} for s in splits},
                       python=sys.version, versions={p: importlib.metadata.version(p) for p in
                            ("numpy", "scipy", "pandas", "scikit-learn", "catboost", "lightgbm")})
        for split in splits:
            np.savez_compressed(work/f"{split.name}.npz", train=split.train, valid=split.valid)
        np.savez_compressed(work/"full.npz", train=np.arange(len(train)), valid=np.empty(0, dtype=int))
        offline = type("Offline", (), {"available": False})()
        research = Research(task, profile, offline)
        controls = []
        # This exact portfolio is recorded before training; no target-name dispatch.
        control_configs = [research.next([]) for _ in range(args.control_trials)]
        alternatives = portfolio(task, profile)
        atomic_json(work/"preregistered_portfolio.json", {"control": [c.to_dict() for c in control_configs],
                    **{name: [c.to_dict() for c in configs] for name, configs in alternatives.items()},
                    "blend_weights": [0, .25, .5, .75, 1.]})
        runner = Runner(work, task, profile, budget, worker_module="novel_worker")
        save("searching")
        print("Исследование: " + str(work), flush=True)

        def run(config, phase, family, source=None):
            allowance = min(args.trial_seconds, budget.remaining()-max(10, budget.finish_margin))
            result = runner.run(config, work/f"{phase}.npz", family+"_"+phase, allowance,
                                refit=phase != "search",
                                fixed_iterations=source["best_iterations"] if source and config.model in
                                {"catboost", "lightgbm", "linear"} else None)
            summary["experiments"].append(result)
            journal.event("experiment_result", family=family, **result)
            print(f'{family}/{phase}: {result["status"]}; {task.metric}={result.get("score")}', flush=True)
            save()
            return result

        def ranked(results):
            return sorted([r for r in results if r.get("status") == "ok"],
                          key=lambda r: r["score"], reverse=task.maximize)

        def predictions(result):
            with np.load(Path(result["directory"])/"predictions.npz", allow_pickle=False) as values:
                return values["valid"], values["test"], values["valid_indices"]

        for config in control_configs:
            controls.append(run(config, "search", "control"))
        if not ranked(controls):
            raise RuntimeError("Контроль не получил рабочую модель")
        control = ranked(controls)[0]
        control_config = ExperimentSpec.from_dict(control["config"], task)
        base_valid, base_test, base_indices = predictions(control)
        write_submission(task, test, base_test, root/"submission.csv")
        chosen = {}
        for family, configs in alternatives.items():
            results = [run(config, "search", family) for config in configs]
            successful = ranked(results)
            if not successful:
                summary["methods"][family] = {"status": "failed", "tested": len(results)}
                continue
            best = successful[0]
            valid, _, indices = predictions(best)
            if not np.array_equal(indices, base_indices):
                raise ValueError("Несовпадение строк сравниваемых моделей")
            alpha, best_blend = 0., control["score"]
            for weight in (.25, .5, .75, 1.):
                score = evaluate(task, y[search.valid], (1-weight)*base_valid+weight*valid)[task.metric]
                if better(task, score, best_blend):
                    alpha, best_blend = weight, score
            chosen[family] = {"result": best, "alpha": alpha}
            summary["methods"][family] = {"status": "search_ok", "search_score": best["score"],
                "search_blend_score": best_blend, "alpha": alpha, "config": best["config"],
                "search_directory": Path(best["directory"]).name,
                "search_seconds": sum(r.get("wall_seconds", 0) for r in results), "tested": len(results)}
        frozen = {"control": {"config": control_config.to_dict(), "best_iterations": control["best_iterations"]},
                  **{name: {"config": v["result"]["config"], "alpha": v["alpha"]} for name, v in chosen.items()}}
        atomic_json(work/"frozen_before_confirmation.json", frozen)
        save("confirming")
        control_confirm = run(control_config, "confirm", "control", control)
        if control_confirm["status"] != "ok":
            raise RuntimeError("Подтверждение контроля не завершено")
        base_valid, base_test, base_indices = predictions(control_confirm)
        summary["methods"]["control"] = {"status": "ok", "search_score": control["score"],
            "confirmation_score": control_confirm["score"], "config": control_config.to_dict(),
            "confirmation_directory": Path(control_confirm["directory"]).name,
            "search_seconds": sum(r.get("wall_seconds", 0) for r in controls)}
        winner, winner_score, winner_alpha = "control", control_confirm["score"], 0.
        for family, selected in chosen.items():
            config = NovelSpec.from_dict(selected["result"]["config"], task)
            result = run(config, "confirm", family, selected["result"])
            record = summary["methods"][family]
            if result["status"] != "ok":
                record["status"] = "confirmation_failed"
                continue
            valid, predicted, indices = predictions(result)
            if not np.array_equal(indices, base_indices):
                raise ValueError("Несовпадение подтверждающих строк")
            alpha = selected["alpha"]
            blended_valid, blended_test = (1-alpha)*base_valid+alpha*valid, (1-alpha)*base_test+alpha*predicted
            blend_metrics = evaluate(task, y[confirm.valid], blended_valid)
            score = blend_metrics[task.metric]
            np.savez_compressed(Path(result["directory"])/"blend_predictions.npz", valid=blended_valid,
                                test=blended_test, valid_indices=indices)
            record.update(status="ok", confirmation_score=result["score"], confirmation_blend_score=score,
                          confirmation_directory=Path(result["directory"]).name,
                          confirmation_seconds=result["wall_seconds"], confirmation_metrics=result["metrics"],
                          blend_metrics=blend_metrics)
            if better(task, score, winner_score):
                winner, winner_score, winner_alpha = family, score, alpha
        summary.update(winner=winner, winner_score=winner_score, winner_alpha=winner_alpha,
                       peak_worker_rss_mb=max(r.get("peak_memory_mb", 0) for r in summary["experiments"]))
        # Final training only after all configurations and comparisons are complete.
        save("refitting")
        full_base = run(control_config, "full", "control", control)
        full_prediction = predictions(full_base)[1] if full_base["status"] == "ok" else None
        full_alternative = None
        if winner != "control" and winner_alpha > 0:
            config = NovelSpec.from_dict(chosen[winner]["result"]["config"], task)
            full_alternative = run(config, "full", winner, chosen[winner]["result"])
            if full_alternative["status"] == "ok" and (full_prediction is not None or winner_alpha == 1):
                novel_prediction = predictions(full_alternative)[1]
                full_prediction = novel_prediction if winner_alpha == 1 else (1-winner_alpha)*full_prediction+winner_alpha*novel_prediction
        if full_prediction is not None:
            write_submission(task, test, full_prediction, root/"submission.csv")
        summary["final_refit_completed"] = full_base["status"] == "ok" and (
            full_alternative is None or full_alternative["status"] == "ok")
        save("completed")
        return 0
    except Exception as error:
        summary["error"] = str(error)[-1800:]
        traceback.print_exc()
        save("failed")
        return 1
    finally:
        timer.cancel()


if __name__ == "__main__":
    raise SystemExit(main())
