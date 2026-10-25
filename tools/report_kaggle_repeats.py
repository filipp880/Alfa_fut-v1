"""Проверка четырёх фиксированных seed на уже исследованном confirm.

Это post-hoc оценка шума обучения, а не независимое подтверждение качества.
Сравниваются два средних предсказаний с весами 1/4, без выбора seed и весов.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tools")]
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from contracts import native
from data_io import encode_target, feature_groups, profile_data
from kaggle_specs import parse_research_spec
from kaggle_xgb import class_metrics
from metrics import evaluate
from report_kaggle import basename, check_score, load_predictions, plan_identity, read_json
from report_unconventional import paired_interval
from task_reader import load_task
from validation import make_splits

SEEDS = (42, 43, 44, 45)
LABELS = ("baseline", "integrated")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_config(config, seed):
    result = copy.deepcopy(config)
    result["seed"] = seed
    if result.get("model") == "augmentation":
        result["base_config"]["seed"] = seed
    return result


def analyze(case):
    return _analyze_seeds(case, SEEDS, ensemble=True)


def analyze_single_seed(case, seed=43):
    if seed != 43:
        raise ValueError("Для одиночной проверки зарегистрирован только seed=43")
    return _analyze_seeds(case, (42, seed), ensemble=False)


def _analyze_seeds(case, seeds, ensemble):
    case = Path(case).resolve(strict=True)
    sources = sorted(case.glob(".kaggle_research/*/summary.json"))
    if not sources:
        raise ValueError("Нет основного Kaggle-исследования")
    source = sources[-1].parent
    original = read_json(sources[-1])
    frozen_path = source/"frozen_before_confirmation.json"
    frozen = read_json(frozen_path)
    if original.get("status") != "completed":
        raise ValueError("Основное исследование не завершено")
    task, train, test, _, dictionary = load_task(case/"task.md")
    if ensemble and (task.classification or task.metric != "rmse"):
        raise ValueError("Эта зарегистрированная проверка предназначена для RMSE-регрессии")
    y, groups = encode_target(task, train), feature_groups(task, train)
    _, _, times = profile_data(task, train, test, dictionary)
    confirm = {s.name: s for s in make_splits(task, y, groups, times)}["confirm"]
    with np.load(source/"confirm.npz", allow_pickle=False) as values:
        if not np.array_equal(values["train"], confirm.train) or not np.array_equal(values["valid"], confirm.valid):
            raise ValueError("Основное confirm-разбиение не воспроизводится")
    candidates = {c["config_key"]: c for c in frozen["candidates"]}
    plans = {label: original[label] for label in LABELS}
    required_keys = list(dict.fromkeys(k for plan in plans.values() for k in plan["members"]))
    for label, plan in plans.items():
        if plan_identity(plan) not in {plan_identity(p) for p in frozen[label+"_plans"]}:
            raise ValueError("Исходный план отсутствует в зафиксированных кандидатах")
        if len(plan["members"]) != len(plan["weights"]) or not np.isclose(sum(plan["weights"]), 1.):
            raise ValueError("Некорректные веса исходного плана")
        if any(w < 0 or not np.isfinite(w) for w in plan["weights"]):
            raise ValueError("Неверный вес исходного плана")
    archive = case/"source.zip"
    manifest = read_json(case/"docker_objects.json")
    if not archive.exists() or sha256(archive) != manifest["archive_sha256"]:
        raise ValueError("Не подтверждён исходный архив snapshot")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Исходный ZIP повреждён")
        for name in bundle.namelist():
            if "/" in name or "\\" in name or bundle.read(name) != (case/name).read_bytes():
                raise ValueError("Snapshot отличается от исходного архива: "+name)

    def check_run(directory, config, fixed, seed):
        request = read_json(directory/"request.json")
        result = read_json(directory/"controller_result.json")
        if not request.get("refit") or request.get("fixed_iterations") != fixed:
            raise ValueError("Повтор допускает early stopping либо изменил число итераций")
        expected = seed_config(config, seed)
        if request["config"] != expected:
            raise ValueError("В конфигурации изменён параметр помимо seed: "+directory.name)
        if parse_research_spec(expected, task).key != result.get("config_key"):
            raise ValueError("Неверный ключ повторной конфигурации")
        split_reference = str(request["split_path"]).replace("\\", "/")
        expected_tail = "/.kaggle_research/"+source.name+"/confirm.npz"
        if not split_reference.endswith(expected_tail):
            raise ValueError("Повтор использует другое разбиение")
        if result.get("status") != "ok" or result.get("n_train") != len(confirm.train):
            raise ValueError("Повтор не завершил обучение на исходном train")
        if result.get("best_iterations") != fixed or result.get("evaluation_used_for_fit", False):
            raise ValueError("Повтор использовал другой префикс модели либо confirm для fit")
        prediction = load_predictions(directory/"predictions.npz", confirm.valid, len(test))
        metrics = evaluate(task, y[confirm.valid], prediction)
        check_score(metrics[task.metric], result.get("score"), directory.name)
        return prediction, result, request

    inputs, predictions, seed_records = [], {}, {}
    original_members = {}
    for key in required_keys:
        config = candidates[key]["config"]
        if config.get("seed") != 42 or (config.get("model") == "augmentation" and config["base_config"].get("seed") != 42):
            raise ValueError("Основной план не соответствует зарегистрированному seed=42")
        matches = []
        for path in source.glob("*/request.json"):
            req = read_json(path)
            if req.get("config") == config and basename(req["split_path"]) == "confirm.npz":
                matches.append(path.parent)
        if len(matches) != 1:
            raise ValueError("Неоднозначное исходное подтверждение: "+key)
        original_members[key] = check_run(matches[0], config, candidates[key]["best_iterations"], 42)
    predictions[42] = {}
    for label, plan in plans.items():
        weighted = sum(w*original_members[k][0] for w, k in zip(plan["weights"], plan["members"]))
        saved = load_predictions(source/plan["predictions_path"], confirm.valid)
        np.testing.assert_array_equal(weighted, saved)
        metrics = evaluate(task, y[confirm.valid], weighted)
        check_score(metrics[task.metric], plan["confirmation_score"], label+" seed42")
        predictions[42][label] = weighted
    original_cost = sum(original_members[k][1].get("wall_seconds", 0.) for k in required_keys)
    seed_records[42] = {"seed": 42, "kind": "Основное заранее зафиксированное исследование",
        "baseline": evaluate(task, y[confirm.valid], predictions[42]["baseline"]),
        "integrated": evaluate(task, y[confirm.valid], predictions[42]["integrated"]),
        "member_confirmation_fit_seconds": original_cost,
        "member_peak_worker_rss_mb": max(original_members[k][1].get("peak_memory_mb", 0.) for k in required_keys)}
    inputs.append({"seed": 42, "summary_path": str(sources[-1]), "summary_sha256": sha256(sources[-1]),
        "member_prediction_directories": {k: basename(original_members[k][1]["directory"]) for k in required_keys},
        "api_note": "Основной поиск использовал общий LLM-пул; исходные confirm-обучения LLM-вызовов не выполняют."})
    repeat_paths = sorted(case.glob(".kaggle_repeat/*/summary.json"))
    by_seed = {}
    for path in repeat_paths:
        item = read_json(path)
        by_seed.setdefault(item.get("seed"), []).append((path, item))
    missing = [seed for seed in seeds[1:] if seed not in by_seed]
    if missing:
        raise ValueError("Пока нет зарегистрированных повторов seed: "+", ".join(map(str, missing)))
    if any(seed not in seeds[1:] for seed in by_seed):
        raise ValueError("Обнаружены дополнительные seed сверх зарегистрированного набора")
    for seed in seeds[1:]:
        if len(by_seed[seed]) != 1:
            raise ValueError("Обнаружен повторный запуск одного seed; нельзя выбирать удачную попытку")
        summary_path, repeat = by_seed[seed][0]
        work = summary_path.parent
        if repeat.get("status") != "completed" or repeat.get("metric") != task.metric:
            raise ValueError("Повтор не завершён либо использует другую метрику")
        if repeat.get("llm_calls") != 0 or repeat.get("llm_tokens") != 0:
            raise ValueError("Повтор использовал LLM")
        if repeat.get("frozen_sha256") != sha256(frozen_path) or repeat.get("confirm_sha256") != sha256(source/"confirm.npz"):
            raise ValueError("Повтор использовал другую фиксацию или разбиение")
        if basename(repeat["source_directory"]) != source.name:
            raise ValueError("Повтор относится к другому основному исследованию")
        if not 0 < repeat.get("elapsed_seconds", float("inf")) <= repeat.get("time_limit_seconds", 0) <= 900:
            raise ValueError("Повтор нарушил зарегистрированный временной лимит")
        if not 1 <= repeat.get("threads", 99) <= 8 or not 128 <= repeat.get("memory_budget_mb", 999999) <= 26000:
            raise ValueError("Недопустимые лимиты процесса повтора")
        snapshot_modules = repeat.get("snapshot_modules", {})
        expected_modules = {"contracts", "data_io", "experiment_store", "kaggle_specs", "metrics", "runtime", "task_reader"}
        if set(snapshot_modules) != expected_modules:
            raise ValueError("Не зарегистрирован полный набор snapshot-модулей")
        for name, value in snapshot_modules.items():
            path = str(value).replace("\\", "/")
            if basename(path) != name+".py" or not (path == "/task/"+name+".py" or Path(path).parent == case):
                raise ValueError("Импортирован модуль вне snapshot: "+name)
        repeated = {r["source_config_key"]: r for r in repeat["experiments"]}
        if set(repeated) != set(required_keys) or len(repeated) != len(repeat["experiments"]):
            raise ValueError("Набор fit не соответствует двум исходным планам")
        members = {}
        for key, stated in repeated.items():
            directory = work/basename(stated["directory"])
            members[key] = check_run(directory, candidates[key]["config"], candidates[key]["best_iterations"], seed)
            check_score(members[key][1]["score"], stated["score"], f"seed{seed} member")
            if stated.get("fixed_iterations") != candidates[key]["best_iterations"]:
                raise ValueError("В summary указано другое число итераций")
        predictions[seed] = {}
        for label, plan in plans.items():
            record = repeat["plans"][label]
            if plan_identity(record) != plan_identity(plan) or record.get("status") != "ok":
                raise ValueError("В повторе изменён состав либо веса исходного плана")
            weighted = sum(w*members[k][0] for w, k in zip(plan["weights"], plan["members"]))
            with np.load(work/record["predictions_path"], allow_pickle=False) as paired:
                np.testing.assert_array_equal(paired["valid_indices"], confirm.valid)
                np.testing.assert_array_equal(paired["y"], y[confirm.valid])
                np.testing.assert_array_equal(paired["original"], predictions[42][label])
                np.testing.assert_array_equal(paired["repeat"], weighted)
            score = evaluate(task, y[confirm.valid], weighted)[task.metric]
            check_score(score, record["repeat_score"], label+f" seed{seed}")
            check_score(original[label]["confirmation_score"], record["original_score"], label+" original")
            predictions[seed][label] = weighted
        resource_path = work/"resource_audit.json"
        resource = read_json(resource_path) if resource_path.exists() else None
        if resource is None and (seed != 43 or not ensemble):
            raise ValueError("Отсутствует обещанный фактический аудит нового повтора seed="+str(seed))
        if resource is not None:
            host = resource.get("HostConfig", resource)
            cpu = host.get("cpu_total", host.get("cpu_limit", host.get("NanoCpus", 0)/1e9))
            memory = host.get("memory_bytes_total", host.get("memory_bytes", host.get("Memory", 0)))
            network = host.get("network", host.get("network_mode", host.get("NetworkMode")))
            if not 0 < cpu <= 8 or not 0 < memory <= 32000000000 or network != "none":
                raise ValueError("Фактические лимиты либо сеть повтора не соответствуют ТЗ")
        fit_seconds = sum(v[1].get("wall_seconds", 0.) for v in members.values())
        check_score(fit_seconds, repeat["fit_seconds"], f"seed{seed} fit time")
        seed_records[seed] = {"seed": seed, "kind": "Фиксированный повтор шума обучения",
            "baseline": evaluate(task, y[confirm.valid], predictions[seed]["baseline"]),
            "integrated": evaluate(task, y[confirm.valid], predictions[seed]["integrated"]),
            "elapsed_seconds": repeat["elapsed_seconds"], "fit_seconds": fit_seconds,
            "peak_worker_rss_mb": max(v[1].get("peak_memory_mb", 0.) for v in members.values()),
            "llm_calls": 0, "llm_tokens": 0, "resource_audit": resource,
            "network_none_verified_by_inspect": resource is not None,
            "resource_note": ("HostConfig снят во время выполнения" if resource is not None else
                "Для seed43 inspect не сохранён до удаления контейнера; отсутствие API подтверждено счётчиками, сетевой режим независимо не подтверждён этим JSON."),
            "snapshot_modules": snapshot_modules}
        inputs.append({"seed": seed, "summary_path": str(summary_path), "summary_sha256": sha256(summary_path),
                       "frozen_sha256": repeat["frozen_sha256"], "confirm_sha256": repeat["confirm_sha256"]})

    gains = []
    direction = 1 if task.maximize else -1
    for seed in seeds:
        record = seed_records[seed]
        gain = direction*(record["integrated"][task.metric]-record["baseline"][task.metric])
        record["absolute_improvement"] = gain
        record["relative_improvement_percent"] = gain/abs(record["baseline"][task.metric])*100
        if not task.classification and task.metric == "rmse":
            record["absolute_rmse_improvement"] = gain
            record["relative_rmse_reduction_percent"] = record["relative_improvement_percent"]
        if task.classification:
            record["per_class"] = {label: class_metrics(task, y[confirm.valid], predictions[seed][label]) for label in LABELS}
        gains.append(gain)
    if not ensemble:
        seed = seeds[-1]
        interval = paired_interval(task, y[confirm.valid], predictions[seed]["baseline"], predictions[seed]["integrated"], groups[confirm.valid], repeats=400)
        return {"status": "completed", "case": str(case), "metric": task.metric, "repeat_seed": seed,
            "interpretation": "Фиксированный повтор шума обучения на уже изученном confirm. Это не новый независимый holdout; лучший seed и новые веса не выбираются.",
            "stopping_rule": "Включены исходный seed42 и единственный повтор seed43; дополнительные seed не подбираются.",
            "inputs": inputs, "seeds": [seed_records[s] for s in seeds],
            "repeat_comparison": {"absolute_improvement": seed_records[seed]["absolute_improvement"],
                "relative_improvement_percent": seed_records[seed]["relative_improvement_percent"],
                "improvement_ci95_conditional": interval,
                "bootstrap_note": "400 парных групповых повторов; интервал условный, без поправки на множественный выбор."},
            "fixed_candidates": {key: candidates[key] for key in required_keys},
            "checks": {"only_seed_changed": True, "weights_preserved": True, "fixed_iterations_preserved": True,
                "confirm_indices_exact": True, "paired_y_exact": True, "metrics_recomputed": True,
                "snapshot_matches_source_zip": True, "repeat_llm_calls_zero": True, "repeat_llm_tokens_zero": True,
                "actual_resource_audit_present": True, "single_repeat_without_seed_selection": True},
            "source_archive_sha256": manifest["archive_sha256"], "frozen_sha256": sha256(frozen_path),
            "confirm_sha256": sha256(source/"confirm.npz")}
    ensembles, ensemble_predictions = {}, {}
    for label in LABELS:
        prediction = sum(predictions[seed][label] for seed in seeds)/4.
        ensemble_predictions[label] = prediction
        scores = [seed_records[seed][label]["rmse"] for seed in seeds]
        ensembles[label] = {"seeds": list(seeds), "weights": [.25]*4,
            "metrics": evaluate(task, y[confirm.valid], prediction),
            "per_seed_rmse_mean": float(np.mean(scores)), "per_seed_rmse_sample_std": float(np.std(scores, ddof=1))}
    improvement = ensembles["baseline"]["metrics"]["rmse"]-ensembles["integrated"]["metrics"]["rmse"]
    interval = paired_interval(task, y[confirm.valid], ensemble_predictions["baseline"], ensemble_predictions["integrated"], groups[confirm.valid], repeats=400)
    try:
        relative_case = str(case.relative_to(ROOT))
    except ValueError:
        relative_case = str(case)
    return {"status": "completed", "case": relative_case, "metric": task.metric,
        "interpretation": "Post-hoc проверка шума обучения на уже изученном confirm. Это не независимое подтверждение, не оценка скрытого теста и не основание подбирать seed/веса.",
        "stopping_rule": "Взяты все и только seeds42/43/44/45; дополнительные seed не подбираются.",
        "inputs": inputs, "seeds": [seed_records[seed] for seed in seeds], "ensembles": ensembles,
        "equal_weight_ensemble_comparison": {"absolute_rmse_improvement": improvement,
            "relative_rmse_reduction_percent": improvement/ensembles["baseline"]["metrics"]["rmse"]*100,
            "improvement_ci95_conditional": interval,
            "bootstrap_note": "400 парных пуассоновских повторов целых групп одинаковых признаков. Условный интервал не учитывает post-hoc выбор, будущий сдвиг и неопределённость распределения seed."},
        "per_seed_gains": {"mean": float(np.mean(gains)), "sample_std": float(np.std(gains, ddof=1)),
                           "minimum": float(np.min(gains)), "maximum": float(np.max(gains)),
                           "positive_seeds": int(np.sum(np.asarray(gains) > 0)), "total_seeds": 4},
        "fixed_candidates": {key: candidates[key] for key in required_keys},
        "checks": {"only_seed_changed": True, "fixed_iterations_preserved": True,
            "confirm_indices_exact": True, "metrics_recomputed": True, "snapshot_matches_source_zip": True,
            "repeat_llm_calls_zero": True, "repeat_llm_tokens_zero": True, "all_four_seeds_included": True},
        "source_archive_sha256": manifest["archive_sha256"], "frozen_sha256": sha256(frozen_path),
        "confirm_sha256": sha256(source/"confirm.npz")}


def main():
    parser = argparse.ArgumentParser(description="Проверка фиксированных четырёх seed без подбора весов")
    parser.add_argument("case")
    parser.add_argument("--single-seed", type=int, choices=[43], help="Проверить исходный seed42 и единственный повтор seed43")
    parser.add_argument("--single-seed43", dest="single_seed", action="store_const", const=43, help=argparse.SUPPRESS)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = analyze_single_seed(args.case, args.single_seed) if args.single_seed else analyze(args.case)
    output = Path(args.output) if args.output else ROOT/"docs"/("KAGGLE_CLASSIFICATION_REPEAT_RESULTS.json" if args.single_seed else "KAGGLE_REPEAT_RESULTS.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(native(result), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for row in result["seeds"]:
        metric = result["metric"]
        print(f'seed{row["seed"]}: контроль {row["baseline"][metric]:.6f}; новая ветвь {row["integrated"][metric]:.6f}; прирост {row["absolute_improvement"]:.6f}')
    if not args.single_seed:
        print(f'Средние предсказаний по четырём seed: контроль {result["ensembles"]["baseline"]["metrics"]["rmse"]:.6f}; '
              f'история {result["ensembles"]["integrated"]["metrics"]["rmse"]:.6f}')
    print("Post-hoc проверка устойчивости; это не независимое подтверждение. Отчёт: "+str(output))


if __name__ == "__main__":
    main()
