"""Полная заранее заданная матрица G6: закрытые независимые контейнеры."""
from pathlib import Path
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
from contracts import ExperimentSpec, native
from contracts import parse_experiment
from data_io import encode_target, profile_data
from experiment_store import atomic_json
from historical_features import HistoricalFeatures, HistorySchema
from history_router import HistoryRouterSpec, suggest_router_schema
from isolated_trials import create_trial, run_phase, _phase_splits
from metrics import evaluate, better
from novel_models import NovelSpec
from package_solution import MODULES
from proposal_models import ProposalSpec
from submission import write_submission
from submission_audit import audit_submission
from task_reader import load_task


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specifications(task, dictionary, profile, frame):
    schema = suggest_router_schema(dictionary, task, profile, frame)
    if not schema:
        raise ValueError("Матрице требуется обоснованная временная схема повторных сущностей")
    cb = ExperimentSpec(depth=8, iterations=2400, learning_rate=.045)
    lgb = ExperimentSpec(model="lightgbm", iterations=2400)
    h0 = HistoryRouterSpec(history_schema=schema)
    columns = [c for c in task.feature_columns if c not in task.categorical_columns and c in profile["missing_train"]]
    return {
        "ET": NovelSpec(forest_depth=24, min_leaf=4, max_features=1.),
        "H0": h0,
        "HM": replace(h0, history_policy="median"),
        "HP": replace(h0, history_policy="missing_pattern"),
        "HS": replace(h0, seed_count=2),
        "CM": ProposalSpec(variant="median", base_config=cb.to_dict(), columns=columns),
        "LH": ProposalSpec(variant="huber", base_config=lgb.to_dict()),
        "LL": ProposalSpec(variant="log_smear", base_config=lgb.to_dict()),
    }, schema


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--trial-seconds", type=int, default=600)
    parser.add_argument("--resume")
    args = parser.parse_args()
    started = time.monotonic()
    run = Path(args.resume).resolve() if args.resume else ROOT / "runs" / "integration" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    if not args.resume: run.mkdir(parents=True)
    print("Матрица интеграции: " + str(run), flush=True)
    task, train, test, _, dictionary = load_task(args.task)
    y = encode_target(task, train)
    full_profile, groups, times = profile_data(task, train, test, dictionary)
    splits = _phase_splits(task, y, groups, times)
    main_phase = "search_main" if task.classification else "search_late"
    search_phases = [p for p in splits if p.startswith("search")]
    profile, _, _ = profile_data(task, train.iloc[splits[main_phase].train], test, dictionary)
    profile["time_column"] = full_profile["time_column"]
    specs, schema = specifications(task, dictionary, profile, train.iloc[splits[main_phase].train])
    summary = {"status": "preparing", "directory": str(run), "task": task.to_dict(),
        "specs": {k: v.to_dict() for k, v in specs.items()}, "schema": schema,
        "protocol_sha256": sha(ROOT / "docs" / "INTEGRATION_PROTOCOL.md"),
        "source_answers": {n: sha(Path("C:/Users/PC/Desktop") / n) for n in ("GPT2.txt", "Q2.txt")},
        "jobs": [], "llm_calls": 0, "selection_note": "Confirm уже исследован ранее; повторная оценка"}
    if args.resume:
        if (run / "frozen_before_confirm.json").exists():
            raise ValueError("Этот режим возобновления разрешён только до открытия confirm")
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        specs = {k: parse_experiment(v, task) for k, v in summary["specs"].items()}
        schema = summary["schema"]
        cases = {k: run / p for k, p in summary["cases"].items()}
        previous_seconds = summary.get("elapsed_seconds", 0)
        started -= previous_seconds
        summary.setdefault("infrastructure_resumes", []).append({"reason": "Собственный Python bytecode cache", "parameters_unchanged": True})
    else:
        cases = {k: create_trial(run / "trials", k, args.task, v.to_dict(), MODULES) for k, v in specs.items()}
        summary["cases"] = {k: str(p.relative_to(run)) for k, p in cases.items()}
    cache = {}
    for job in summary["jobs"]:
        with np.load(run / job["directory"] / "predictions.npz", allow_pickle=False) as values:
            np.testing.assert_array_equal(values["valid_indices"], splits[job["phase"]].valid)
            cache[(job["label"], job["phase"])] = (values["valid"].copy(), values["test"].copy())
    def save(status=None):
        if status: summary["status"] = status
        summary["elapsed_seconds"] = time.monotonic() - started
        atomic_json(run / "summary.json", summary)
    def execute(label, phase, count=None, case=None):
        if (label, phase) in cache:
            return next(j for j in summary["jobs"] if j["label"] == label and j["phase"] == phase)
        location = (case or cases[label]) / "phases" / phase
        if location.exists() and (location / "host_failure.json").exists():
            location.resolve().relative_to(run)
            failed = run / "infrastructure_failures" / (label + "_" + phase)
            failed.parent.mkdir(exist_ok=True)
            if failed.exists(): raise ValueError("Повторный отказ инфраструктуры требует отдельного аудита")
            location.rename(failed)
        if time.monotonic() - started >= 10800 - 30:
            raise TimeoutError("Бюджет всей матрицы исчерпан")
        result = run_phase(case or cases[label], phase, time_limit=min(1200, 10800 - (time.monotonic()-started)),
            trial_seconds=args.trial_seconds, fixed_iterations=count, trusted_train_path=task.train_path)
        with np.load(result["prediction_path"], allow_pickle=False) as values:
            np.testing.assert_array_equal(values["valid_indices"], splits[phase].valid)
            prediction = (values["valid"].copy(), values["test"].copy())
        score = evaluate(task, y[splits[phase].valid], prediction[0])[task.metric] if len(splits[phase].valid) else None
        if phase.startswith("search") and not np.isclose(score, result["summary"]["score"], atol=1e-10, rtol=1e-10):
            raise ValueError("Поисковая метрика worker не воспроизвелась")
        job = {"label": label, "phase": phase, "directory": str(Path(result["phase_directory"]).relative_to(run)),
            "score": score, "best_iterations": result["summary"]["best_iterations"],
            "host_seconds": result["host_elapsed_seconds"], "worker": result["summary"]["worker_result"],
            "prediction_sha256": sha(result["prediction_path"]), "label_blinded": result["summary"]["label_blinded"]}
        summary["jobs"].append(job)
        cache[(label, phase)] = prediction
        save()
        print(f'{label}/{phase}: {task.metric}={score}, count={job["best_iterations"]}', flush=True)
        return job
    def mask(phase):
        h = HistoricalFeatures(HistorySchema(**schema)).fit(train.iloc[splits[phase].train][task.feature_columns], np.zeros(len(splits[phase].train)))
        return tuple(h.transform(frame[task.feature_columns])["__history_known"].to_numpy(dtype=bool)
                     for frame in (train.iloc[splits[phase].valid], test))
    def compose(phase, weights=None, prefix=""):
        out = {label: cache[(prefix+label, phase)] for label in specs if (prefix+label, phase) in cache}
        known = mask(phase)
        for label in ("CM", "LH", "LL"):
            if label not in out or "H0" not in out: continue
            for local in (False, True):
                name = ("unknown_" if local else "blend_") + label
                alpha = (weights or {}).get(name, 1.)
                if local:
                    out[name] = tuple(np.where(known[i], out["H0"][i], (1-alpha)*out["ET"][i]+alpha*out[label][i]) for i in (0, 1))
                else:
                    out[name] = tuple((1-alpha)*out["H0"][i]+alpha*out[label][i] for i in (0, 1))
        return out
    try:
        save("searching")
        counts = {}
        for phase in search_phases:
            for label in specs:
                job = execute(label, phase)
                if phase == main_phase: counts[label] = job["best_iterations"]
        indices = np.concatenate([splits[p].valid for p in search_phases]); pooled_y = y[indices]
        base_phase = {p: compose(p) for p in search_phases}
        base_pred = np.concatenate([base_phase[p]["H0"][0] for p in search_phases])
        base_score = evaluate(task, pooled_y, base_pred)[task.metric]
        base_folds = {p: evaluate(task, y[splits[p].valid], base_phase[p]["H0"][0])[task.metric] for p in search_phases}
        recipes = {}; weights = {}; winner = "H0"; winning_score = base_score
        labels = ["H0", "HM", "HP", "HS", "blend_CM", "blend_LH", "blend_LL", "unknown_CM", "unknown_LH", "unknown_LL"]
        for label in labels:
            choices = (0., .25, .5, .75, 1.) if label.startswith(("blend_", "unknown_")) else (1.,)
            chosen = None
            for alpha in choices:
                values = {p: compose(p, {label: alpha})[label] for p in search_phases}
                pooled = np.concatenate([values[p][0] for p in search_phases])
                score = evaluate(task, pooled_y, pooled)[task.metric]
                fold_scores = {p: evaluate(task, y[splits[p].valid], values[p][0])[task.metric] for p in search_phases}
                eligible = all(fold_scores[p] <= base_folds[p] + .001*max(1, abs(base_folds[p])) for p in search_phases)
                if chosen is None or better(task, score, chosen["search_score"], .001):
                    chosen = {"weight": alpha, "search_score": score, "search_by_fold": fold_scores, "eligible": eligible}
            recipes[label] = chosen; weights[label] = chosen["weight"]
            if chosen["eligible"] and better(task, chosen["search_score"], winning_score, .001):
                winner, winning_score = label, chosen["search_score"]
        frozen = {"recipes": recipes, "weights": weights, "winner_by_search": winner, "counts": counts,
            "specs": summary["specs"], "baseline_search_score": base_score, "baseline_by_fold": base_folds,
            "source_manifests": {k: sha(p / "manifest.json") for k, p in cases.items()}}
        atomic_json(run / "frozen_before_confirm.json", frozen)
        summary.update(frozen=frozen, winner_by_search=winner)
        save("confirming")
        for label in specs: execute(label, "confirm", counts[label])
        outputs = compose("confirm", weights)
        scores = {k: evaluate(task, y[splits["confirm"].valid], p[0])[task.metric] for k, p in outputs.items()}
        selected = "H0" if better(task, scores["H0"], scores["ET"], .001) else "ET"
        if winner != "H0" and selected == "H0" and better(task, scores[winner], scores["H0"], .001): selected = winner
        summary.update(confirm_scores=scores, selected=selected)
        # Условный повтор заранее выбранного победителя; weights/counts не меняются.
        needed = {"ET", "H0"}
        if winner in specs: needed.add(winner)
        elif winner.startswith(("blend_", "unknown_")): needed.add(winner.split("_", 1)[1])
        save("repeating")
        for label in sorted(needed):
            spec = replace(specs[label], seed=43)
            if spec.model == "proposal": spec = replace(spec, base_config={**spec.base_config, "seed": 43})
            source_manifest = json.loads((cases[label] / "manifest.json").read_text(encoding="utf-8"))
            exact_sources = [cases[label] / name for name in source_manifest["source_files"]]
            case = create_trial(run / "repeats", "repeat43_"+label, args.task, spec.to_dict(), exact_sources)
            execute("repeat43_"+label, "confirm", counts[label], case)
        repeated = compose("confirm", weights, prefix="repeat43_")
        summary["repeat43"] = {k: evaluate(task, y[splits["confirm"].valid], repeated[k][0])[task.metric] for k in {"ET", "H0", winner}}
        save("refitting")
        for label in specs: execute(label, "full", counts[label])
        full = compose("full", weights)
        outdir = run / "submissions"; outdir.mkdir()
        for label, predictions in full.items():
            output = outdir / (label + ".csv")
            write_submission(task, test, predictions[1], output)
            summary.setdefault("submission_audit", {})[label] = audit_submission(task, test, output)
        summary["selected_output"] = str((outdir / (selected + ".csv")).relative_to(run))
        save("completed")
        print(json.dumps({"run": str(run), "selected": selected, "scores": scores, "repeat43": summary["repeat43"]}, ensure_ascii=False), flush=True)
    except Exception as error:
        summary["error"] = str(error)[-1800:]; save("failed"); raise


if __name__ == "__main__": main()
