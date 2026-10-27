"""Полное воспроизведение интегрированного оператора против независимых ветвей."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from experiment_store import atomic_json
from isolated_trials import create_trial, run_phase
from package_solution import MODULES
from report_integration import known_masks, rmse
from task_reader import load_task
from submission import write_submission
from submission_audit import audit_submission


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("run"); args = parser.parse_args()
    run = Path(args.run).resolve(); s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if s["status"] != "completed" or s["selected"] != "unknown_LH": raise ValueError("Нужен подтверждённый заранее выбранный рецепт")
    f = s["frozen"]; task, train, test, *_ = load_task(s["task"]["task_path"]); y = train[task.target].to_numpy(dtype=float)
    config = {**f["specs"]["H0"], "fallback": "forest_huber", "fallback_weight": f["weights"]["unknown_LH"], "fallback_iterations": f["counts"]["LH"]}
    old_case = run / "release_check" / "HF"
    sources = MODULES
    if old_case.exists():
        manifest = json.loads((old_case/"manifest.json").read_text(encoding="utf-8"))
        sources = [old_case/n for n in manifest["source_files"]]
    case = create_trial(run / "release_check", "HF_verified", task.task_path, config, sources)
    results = {}; full_prediction = None
    for phase in ("search_early", "search_late", "confirm", "full"):
        reference = run / s["cases"]["H0"] / "phases" / phase
        source = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
        count = source["best_iterations"] if phase.startswith("search") else f["counts"]["H0"]
        result = run_phase(case, phase, time_limit=1200, trial_seconds=600,
            fixed_iterations=None if phase.startswith("search") else count, trusted_train_path=task.train_path)
        if result["summary"]["best_iterations"] != count: raise ValueError("Число итераций поиска не воспроизвелось")
        with np.load(result["split_path"]) as data: tr, va = data["train"], data["valid"]
        pieces = {}
        for label in ("H0", "ET", "LH"):
            with np.load(run / s["cases"][label] / "phases" / phase / "predictions.npz") as data:
                np.testing.assert_array_equal(data["valid_indices"], va)
                pieces[label] = (data["valid"].copy(), data["test"].copy())
        masks = known_masks(train, test, s["schema"], tr, va); alpha = config["fallback_weight"]
        expected = tuple(np.where(masks[i], pieces["H0"][i], (1-alpha)*pieces["ET"][i]+alpha*pieces["LH"][i]) for i in (0,1))
        with np.load(result["prediction_path"]) as data:
            actual = (data["valid"].copy(), data["test"].copy()); np.testing.assert_array_equal(data["valid_indices"], va)
        for a,b in zip(actual,expected): np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-10)
        maximum = max(float(np.max(np.abs(a-b))) if len(a) else 0. for a,b in zip(actual,expected))
        results[phase] = {"rmse": rmse(y[va],actual[0]) if len(va) else None,"maximum_prediction_difference": maximum,
            "history_iterations": count,"history_early_stopping": phase.startswith("search"),
            "fixed_huber_iterations": config["fallback_iterations"],
            "worker_seconds": result["host_elapsed_seconds"],"component_fits": result["summary"]["worker_result"]["component_fits"]}
        full_prediction = actual[1]
        print(f'Цельный HF/{phase}: RMSE={results[phase]["rmse"]}, maxdiff={maximum}',flush=True)
    output = run / "release_check" / "HF.csv"; write_submission(task,test,full_prediction,output)
    report = {"case":str(case.relative_to(ROOT)),"source_manifest_sha256":hashlib.sha256((case/"manifest.json").read_bytes()).hexdigest(),
        "config":config,"phases":results,"submission":audit_submission(task,test,output),
        "check":"Поиски, confirm и полный refit совпали с независимо обученными компонентами; не новый blindtest"}
    atomic_json(run/"release_operator_verification.json",report)


if __name__=="__main__": main()
