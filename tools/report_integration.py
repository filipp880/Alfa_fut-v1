"""Независимые индексы, ручной RMSE, freeze и аудит закрытых вариантов G6."""
from pathlib import Path
import argparse
import csv
from io import StringIO
import hashlib
import json
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[key] = "1"
import numpy as np
import pandas as pd
from task_reader import load_task
from submission import make_submission
from report_unconventional import paired_interval


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rmse(y, p): return float(np.sqrt(np.sum(np.square(np.asarray(p)-np.asarray(y)), dtype=np.float64)/len(y)))
def close(a, b, label):
    if b is None or not np.isclose(a, b, rtol=1e-10, atol=1e-10): raise ValueError(f"Не воспроизводится {label}: {a} != {b}")


def known_masks(train, test, schema, tr, va):
    """Независимая проверка наличия прошлых X, без исторических target-агрегатов."""
    from numbers import Real
    def key(row):
        out = []
        for value in row:
            if pd.isna(value): return None
            if isinstance(value, (bool, np.bool_)): out.append(("bool", bool(value)))
            elif isinstance(value, Real):
                if not np.isfinite(float(value)): return None
                out.append(("number", float(value)))
            else: out.append(("string", str(value)))
        return tuple(out)
    columns = schema["entity_columns"]; time_column = schema["time_column"]
    histories = {}
    for row, period in zip(train.iloc[tr][columns].itertuples(index=False, name=None), train.iloc[tr][time_column]):
        k = key(row)
        if k is not None and np.isfinite(float(period)): histories.setdefault(k, []).append(float(period))
    result = []
    for frame in (train.iloc[va], test):
        result.append(np.array([sum(t < float(period) for t in histories.get(key(row), ())) >= schema.get("min_count", 1)
            for row, period in zip(frame[columns].itertuples(index=False, name=None), frame[time_column])], dtype=bool))
    return result


def analyze(run):
    run = Path(run).resolve(); s = read(run / "summary.json"); frozen_path = run / "frozen_before_confirm.json"
    if s["status"] != "completed": raise ValueError("Матрица ещё не завершена")
    f = read(frozen_path)
    if f != s["frozen"]: raise ValueError("Изменены frozen-параметры")
    task, train, test, _, dictionary = load_task(s["task"]["task_path"])
    y = train[task.target].to_numpy(dtype=float)
    cache = {}; splits = {}; jobs = []; count = 0
    raw_rows = list(csv.reader(Path(task.train_path).open(encoding="utf-8-sig", newline="")))
    target_column = raw_rows[0].index(task.target)
    for label, relative in s["cases"].items():
        case = run / relative; manifest = read(case / "manifest.json")
        if sha(case / "manifest.json") != f["source_manifests"][label]: raise ValueError("Изменён source manifest")
        for name, record in manifest["immutable_files"].items():
            if sha(case / name) != record["sha256"]: raise ValueError("Изменён источник/вход " + name)
        if read(case / "config.json") != f["specs"][label]: raise ValueError("Изменён рецепт")
    for job in s["jobs"]:
        directory = run / job["directory"]; summary = read(directory / "summary.json"); request = read(directory / "request.json")
        label, phase = job["label"], job["phase"]; original_label = label.removeprefix("repeat43_")
        case = directory.parent.parent; manifest = read(case / "manifest.json")
        primary_manifest = read(run / s["cases"][original_label] / "manifest.json")
        if manifest["source_files"] != primary_manifest["source_files"]: raise ValueError("Другой набор исходников повтора")
        for name in manifest["source_files"]:
            if manifest["immutable_files"][name] != primary_manifest["immutable_files"][name] or sha(case/name) != manifest["immutable_files"][name]["sha256"]:
                raise ValueError("Исходники повтора отличаются от основного снимка")
        with np.load(directory / "split.npz", allow_pickle=False) as data: tr, va = data["train"], data["valid"]
        for a in (tr, va):
            if a.ndim != 1 or a.dtype.kind not in "iu" or len(np.unique(a)) != len(a): raise ValueError("Неверные индексы")
        if np.intersect1d(tr, va).size: raise ValueError("Пересечение train/valid")
        if len(va) and train.iloc[tr][s["schema"]["time_column"]].max() >= train.iloc[va][s["schema"]["time_column"]].min():
            raise ValueError("Нарушена временная последовательность")
        if phase in splits:
            for a, b in zip(splits[phase], (tr, va)): np.testing.assert_array_equal(a, b)
        else: splits[phase] = (tr, va)
        if phase.startswith("search"):
            confirm = np.load(case / "splits" / "confirm.npz")["valid"]
            if np.intersect1d(np.r_[tr, va], confirm).size: raise ValueError("Confirm доступен поиску")
        if phase in {"confirm", "full"}:
            if not request["refit"] or request["fixed_iterations"] != f["counts"][original_label]: raise ValueError("Не frozen refit")
            if frozen_path.stat().st_mtime_ns > (directory / "frozen_request.json").stat().st_mtime_ns: raise ValueError("Fit раньше freeze")
            if summary["worker_result"].get("evaluation_used_for_fit", False): raise ValueError("Confirm early stopping")
            if summary["best_iterations"] != f["counts"][original_label]: raise ValueError("Число итераций не соблюдено")
        config = json.loads(json.dumps(f["specs"][original_label]))
        if label.startswith("repeat43_"):
            config["seed"] = 43
            if "base_config" in config: config["base_config"]["seed"] = 43
        if request["config"] != config: raise ValueError("Параметры изменены после выбора")
        audit = read(directory / "runtime_isolation_attestation.json"); masking = read(directory / "masking_audit.json")
        if not (audit["attested_before_fit"] and audit["network"] == "none" and audit["cpu_total"] <= 8
                and audit["memory_bytes_total"] <= 32000000000 and audit["memory_swap_bytes"] == audit["memory_bytes_total"]
                and audit["image_checkout_hidden"] and audit["no_socket_or_home_mounts"]): raise ValueError("Изоляция нарушена")
        binds = audit["actual_bind_mounts"]
        if len(binds) != 1 or binds[0]["destination"] != "/task": raise ValueError("Посторонний mount")
        allowed = np.union1d(tr, va) if phase.startswith("search") else tr
        if hashlib.sha256(np.asarray(allowed, dtype="<i8").tobytes()).hexdigest() != masking["allowed_label_indices_sha256"]:
            raise ValueError("Изменена разрешённая область меток")
        # Восстанавливаем masked-input из raw строк независимо от host helper.
        permitted = set(allowed.tolist()); replacement = raw_rows[int(tr[0])+1][target_column]
        stream = StringIO(newline=""); writer = csv.writer(stream, lineterminator="\n"); writer.writerow(raw_rows[0])
        for i, source_row in enumerate(raw_rows[1:]):
            row = list(source_row)
            if i not in permitted: row[target_column] = replacement
            writer.writerow(row)
        if hashlib.sha256(stream.getvalue().encode("utf-8")).hexdigest() != masking["masked_train_sha256"]:
            raise ValueError("Маскированный CSV не воспроизводится")
        if phase == "confirm" and (summary["score"] is not None or not masking["label_blinded"]): raise ValueError("Контрольные метки раскрыты")
        path = directory / "predictions.npz"
        if sha(path) != job["prediction_sha256"]: raise ValueError("Изменены прогнозы")
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["valid_indices"], va)
            v, t = data["valid"].copy(), data["test"].copy()
        if len(v) != len(va) or len(t) != len(test) or not np.isfinite(v).all() or not np.isfinite(t).all(): raise ValueError("Неверные прогнозы")
        if len(va): close(rmse(y[va], v), job["score"], label+phase)
        if summary["worker_result"]["n_train"] != len(tr): raise ValueError("Неверное число fit-строк")
        cache[(label, phase)] = (v, t)
        fits = {"ET": 1, "H0": 2, "HM": 2, "HP": 2, "HS": 4, "CM": 1, "LH": 1, "LL": 2}[original_label]
        count += fits; jobs.append({"label": label, "phase": phase, "fits": fits, "seconds": job["host_seconds"]})
    def compose(phase, prefix=""):
        out = {label: cache[(prefix+label, phase)] for label in f["specs"] if (prefix+label, phase) in cache}
        known = known_masks(train, test, s["schema"], *splits[phase])
        for label in ("CM", "LH", "LL"):
            if label not in out: continue
            for name in ("blend_"+label, "unknown_"+label):
                alpha = f["weights"][name]
                out[name] = tuple(np.where(known[i], out["H0"][i], (1-alpha)*out["ET"][i]+alpha*out[label][i])
                    if name.startswith("unknown") else (1-alpha)*out["H0"][i]+alpha*out[label][i] for i in (0, 1))
        return out
    early, late, confirm, full = [compose(p) for p in ("search_early", "search_late", "confirm", "full")]
    pooled_y = np.r_[y[splits["search_early"][1]], y[splits["search_late"][1]]]
    methods = {}; best = "H0"; best_score = f["baseline_search_score"]
    for label, record in f["recipes"].items():
        score = rmse(pooled_y, np.r_[early[label][0], late[label][0]]); close(score, record["search_score"], label+" pooled")
        for phase, output in (("search_early", early), ("search_late", late)):
            close(rmse(y[splits[phase][1]], output[label][0]), record["search_by_fold"][phase], label+phase)
        eligible = all(record["search_by_fold"][p] <= f["baseline_by_fold"][p] + .001*max(1,abs(f["baseline_by_fold"][p]))
                       for p in ("search_early", "search_late"))
        if eligible != record["eligible"]: raise ValueError("Изменён критерий худшего фолда")
        if label.startswith(("blend_", "unknown_")):
            component = label.split("_",1)[1]; selected_alpha = None; replay_score = None
            for alpha in (0.,.25,.5,.75,1.):
                parts = []
                for phase, output in (("search_early", early), ("search_late", late)):
                    known = known_masks(train,test,s["schema"],*splits[phase])[0]
                    p = np.where(known,output["H0"][0],(1-alpha)*output["ET"][0]+alpha*output[component][0]) if label.startswith("unknown_") else (1-alpha)*output["H0"][0]+alpha*output[component][0]
                    parts.append(p)
                value = rmse(pooled_y,np.concatenate(parts))
                if replay_score is None or replay_score-value > .001*max(1,abs(replay_score)):
                    selected_alpha,replay_score = alpha,value
            if selected_alpha != record["weight"]: raise ValueError("Вес выбран не по поиску")
        cscore = rmse(y[splits["confirm"][1]], confirm[label][0]); close(cscore, s["confirm_scores"][label], label+" confirm")
        if record["eligible"] and best_score-score > .001*max(1, abs(best_score)): best, best_score = label, score
        groups = pd.util.hash_pandas_object(train.iloc[splits["confirm"][1]][task.feature_columns], index=False).to_numpy()
        methods[label] = {**record, "confirm_rmse": cscore, "improvement_over_H0": s["confirm_scores"]["H0"]-cscore,
            "ci95_conditional_vs_H0": paired_interval(task, y[splits["confirm"][1]], confirm["H0"][0], confirm[label][0], groups, 400)}
    if best != f["winner_by_search"]: raise ValueError("Выбран другой поисковый победитель")
    c = s["confirm_scores"]
    expected_selected = "H0" if c["ET"]-c["H0"] > .001*max(1,abs(c["ET"])) else "ET"
    if best != "H0" and expected_selected == "H0" and c["H0"]-c[best] > .001*max(1,abs(c["H0"])): expected_selected=best
    if expected_selected != s["selected"]: raise ValueError("Изменено правило подтверждения")
    repeated = compose("confirm", "repeat43_")
    for label, score in s["repeat43"].items(): close(rmse(y[splits["confirm"][1]], repeated[label][0]), score, label+" repeat43")
    for label, audit in s["submission_audit"].items():
        path = run / "submissions" / (label + ".csv")
        if sha(path) != audit["sha256"]: raise ValueError("Изменён CSV")
        expected = make_submission(task, test, full[label][1])
        actual = pd.read_csv(path, dtype={c: str for c in task.id_columns}, keep_default_na=False)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-12)
    known = known_masks(train, test, s["schema"], *splits["confirm"])[0]; target = y[splits["confirm"][1]]
    segments = {name: {"rows": int(mask.sum()), **{k: rmse(target[mask], confirm[k][0][mask]) for k in ("ET", "H0", best)}}
                for name, mask in (("known", known), ("unknown", ~known))}
    superseded_fits = sum({"ET":1,"H0":2,"LH":1}.get(j["label"].removeprefix("repeat43_"),0) for j in s.get("superseded_repeat_jobs",[]))
    return {"directory": str(run.relative_to(ROOT)), "selected": s["selected"], "worker_jobs": len(jobs), "ml_fits": count,
        "superseded_repeat_jobs": len(s.get("superseded_repeat_jobs",[])), "all_executed_ml_fits": count+superseded_fits,
        "methods": methods, "control_ET": s["confirm_scores"]["ET"], "control_H0": s["confirm_scores"]["H0"],
        "selected_score": s["confirm_scores"][s["selected"]], "repeat43": s["repeat43"], "segments": segments,
        "elapsed_seconds": s["elapsed_seconds"], "jobs": jobs, "infrastructure_resumes": s.get("infrastructure_resumes", []),
        "checks": {"raw_X_and_allowed_y_preserved": True, "true_confirm_hidden": True, "manual_metrics": True,
            "indices_disjoint_and_chronological": True, "separate_mounts": True, "frozen_configs_and_counts": True,
            "complete_refits_and_csv": True, "source_manifests_intact": True,"repeat_source_identical":True}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("run"); args = parser.parse_args()
    result = analyze(args.run)
    output = ROOT / "docs" / "INTEGRATION_RESULTS.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"selected": result["selected"], "score": result["selected_score"], "jobs": result["worker_jobs"], "ML_fits": result["ml_fits"]}, ensure_ascii=False))


if __name__ == "__main__": main()
