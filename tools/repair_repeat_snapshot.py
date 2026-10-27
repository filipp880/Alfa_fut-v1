"""Однократное восстановление повторов из точного снимка, без изменения рецепта."""
from pathlib import Path
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from experiment_store import atomic_json
from isolated_trials import create_trial, run_phase
from task_reader import load_task
from report_integration import known_masks, rmse


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("run"); args = parser.parse_args()
    run = Path(args.run).resolve()
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if s["status"] != "completed": raise ValueError("Сначала завершить исходную матрицу")
    f = s["frozen"]
    task, train, test, *_ = load_task(s["task"]["task_path"])
    y = train[task.target].to_numpy(dtype=float)
    winner = f["winner_by_search"]
    labels = {"ET", "H0"}
    labels.add(winner if winner in f["specs"] else winner.split("_", 1)[1])
    snapshot = run / "summary_before_repeat_source_repair.json"
    if snapshot.exists(): raise ValueError("Повторное исправление запрещено")
    snapshot.write_bytes((run / "summary.json").read_bytes())
    cache = {}; replacement_jobs = []
    for label in sorted(labels):
        primary = run / s["cases"][label]
        manifest = json.loads((primary / "manifest.json").read_text(encoding="utf-8"))
        config = deepcopy(f["specs"][label]); config["seed"] = 43
        if "base_config" in config: config["base_config"]["seed"] = 43
        sources = [primary / n for n in manifest["source_files"]]
        case = create_trial(run / "repeats_fixed", "repeat43_"+label, task.task_path, config, sources)
        cm = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
        for name in manifest["source_files"]:
            if cm["immutable_files"][name] != manifest["immutable_files"][name]: raise ValueError("Снимок повторно изменён")
        result = run_phase(case, "confirm", time_limit=1200, trial_seconds=600,
            fixed_iterations=f["counts"][label], trusted_train_path=task.train_path)
        with np.load(result["split_path"]) as split: tr, va = split["train"], split["valid"]
        with np.load(result["prediction_path"]) as data:
            np.testing.assert_array_equal(data["valid_indices"], va)
            prediction = data["valid"].copy()
        score = rmse(y[va], prediction); cache[label] = prediction
        replacement_jobs.append({"label": "repeat43_"+label, "phase": "confirm",
            "directory": str(Path(result["phase_directory"]).relative_to(run)), "score": score,
            "best_iterations": result["summary"]["best_iterations"], "host_seconds": result["host_elapsed_seconds"],
            "worker": result["summary"]["worker_result"], "label_blinded": True,
            "prediction_sha256": hashlib.sha256(Path(result["prediction_path"]).read_bytes()).hexdigest()})
        print(f'Точный снимок repeat43 {label}: RMSE={score}', flush=True)
    known = known_masks(train, test, s["schema"], tr, va)[0]
    if winner in cache: p = cache[winner]
    else:
        component = winner.split("_", 1)[1]; alpha = f["weights"][winner]
        p = np.where(known, cache["H0"], (1-alpha)*cache["ET"]+alpha*cache[component]) if winner.startswith("unknown_") else (1-alpha)*cache["H0"]+alpha*cache[component]
    s["superseded_repeat_jobs"] = [j for j in s["jobs"] if j["label"].startswith("repeat43_")]
    s["jobs"] = [j for j in s["jobs"] if not j["label"].startswith("repeat43_")] + replacement_jobs
    s["repeat43_before_source_repair"] = s["repeat43"]
    s["repeat43"] = {"ET": rmse(y[va], cache["ET"]), "H0": rmse(y[va], cache["H0"]), winner: rmse(y[va], p)}
    s["repeat_source_repair"] = {"reason": "Сохранённый исходный снимок вместо редактируемого checkout", "parameters_unchanged": True,
        "superseded_jobs_preserved": True, "no_selection_changed": True, "utc": datetime.now(timezone.utc).isoformat()}
    if "error" in s: s["recovered_infrastructure_error"] = s.pop("error")
    atomic_json(run / "summary.json", s)
    print(json.dumps(s["repeat43"], ensure_ascii=False))


if __name__ == "__main__": main()
