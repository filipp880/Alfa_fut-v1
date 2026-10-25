"""Независимый пересчёт Kaggle/AIDE-исследования по сохранённым предсказаниям.

Интервалы условны на уже исследованном фолде: они не учитывают будущий сдвиг
и множественный выбор гипотез. Равные лимиты попыток не означают одинаковое
фактическое время вычислений; отчёт сохраняет обе величины.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"tools"))
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from contracts import native
from data_io import encode_target, feature_groups, profile_data
from kaggle_specs import parse_research_spec
from kaggle_xgb import class_metrics
from metrics import better, evaluate
from report_unconventional import paired_interval
from task_reader import load_task
from validation import make_splits

REQUIREMENTS_SHA256 = "9173b0388532fbd2a709e4f4b3029e3b9ccab685fce43a85261ffa1ba95efbe6"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def basename(path):
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def check_score(actual, expected, label):
    if expected is None or not np.isclose(actual, expected, rtol=1e-10, atol=1e-10):
        raise ValueError(f"Метрика не воспроизводится: {label}: {actual} != {expected}")


def load_predictions(path, expected_indices, test_rows=None):
    with np.load(path, allow_pickle=False) as values:
        indices, prediction = values["valid_indices"], values["valid"]
        if not np.array_equal(indices, expected_indices):
            raise ValueError("Порядок либо состав строк оценки отличается: "+str(path))
        if len(prediction) != len(indices) or not np.isfinite(prediction).all():
            raise ValueError("Некорректные предсказания: "+str(path))
        if test_rows is not None:
            if "test" not in values or len(values["test"]) != test_rows or not np.isfinite(values["test"]).all():
                raise ValueError("Некорректные тестовые предсказания: "+str(path))
        return prediction.copy()


def plan_identity(plan):
    return (tuple(plan["members"]), tuple(plan["weights"]))


def analyze(case):
    case = Path(case).resolve()
    paths = sorted(case.glob(".kaggle_research/*/summary.json"))
    if not paths:
        raise ValueError("Нет журнала Kaggle-исследования: "+str(case))
    summary_path = paths[-1]
    work, summary = summary_path.parent, read_json(summary_path)
    if summary.get("status") != "completed":
        raise ValueError("Исследование не завершено: "+str(summary.get("status")))
    task, train, test, _, dictionary = load_task(case/"task.md")
    y, groups = encode_target(task, train), feature_groups(task, train)
    _, _, times = profile_data(task, train, test, dictionary)
    expected_splits = {s.name: s for s in make_splits(task, y, groups, times)}
    expected_splits["full"] = type("FullSplit", (), {"train": np.arange(len(train)), "valid": np.empty(0, dtype=int)})()
    for phase, split in expected_splits.items():
        with np.load(work/(phase+".npz"), allow_pickle=False) as values:
            if not np.array_equal(values["train"], split.train) or not np.array_equal(values["valid"], split.valid):
                raise ValueError("Сохранённое разбиение не воспроизводится: "+phase)
    frozen_path = work/"frozen_before_confirmation.json"
    frozen, registered = read_json(frozen_path), read_json(work/"preregistered.json")
    frozen_candidates = {c["config_key"]: c for c in frozen["candidates"]}
    if len(frozen_candidates) != len(frozen["candidates"]):
        raise ValueError("Повторяющиеся кандидаты в зафиксированном отборе")
    requests = sorted(work.glob("*/request.json"))
    if not requests:
        raise ValueError("Нет запросов исполнителя")
    request_map, checked_experiments = {}, []
    confirm_by_key, search_by_directory = {}, {}
    checks = {"splits_reproduced": True, "all_prediction_indices_exact": True,
              "all_metrics_recomputed": True, "frozen_before_confirmation": True,
              "confirmation_refit_without_early_stopping": True, "fixed_iterations_match_frozen": True}
    for request_path in requests:
        request = read_json(request_path)
        directory = request_path.parent
        phase = basename(request["split_path"]).removesuffix(".npz")
        if phase not in expected_splits:
            raise ValueError("Исполнитель использует неизвестное разбиение")
        spec = parse_research_spec(request["config"], task)
        key = spec.key
        request_map[directory.name] = request
        if phase == "search" and (work/"preregistered.json").stat().st_mtime_ns > request_path.stat().st_mtime_ns:
            raise ValueError("Первичная регистрация гипотез произошла после начала поиска")
        if phase != "search":
            if not request.get("refit"):
                raise ValueError("На подтверждении/refit разрешена ранняя остановка: "+directory.name)
            source = frozen_candidates.get(key)
            if source is None or source["config"] != request["config"]:
                raise ValueError("Подтверждается не зафиксированная конфигурация: "+directory.name)
            if request.get("fixed_iterations") != source["best_iterations"]:
                raise ValueError("Число итераций отличается от поиска: "+directory.name)
            if phase == "confirm" and frozen_path.stat().st_mtime_ns > request_path.stat().st_mtime_ns:
                raise ValueError("Конфигурации записаны после начала подтверждения")
        result_path = directory/"controller_result.json"
        result = read_json(result_path) if result_path.exists() else {"status": "skipped", "reason": "Исполнитель не стартовал"}
        item = {"directory": directory.name, "phase": phase, "config_key": key, "model": spec.model,
                "status": result.get("status"), "wall_seconds": result.get("wall_seconds", 0.),
                "peak_memory_mb": result.get("peak_memory_mb", 0.), "best_iterations": result.get("best_iterations"),
                "fixed_iterations": request.get("fixed_iterations"), "refit": request.get("refit", False),
                "reason": result.get("error", result.get("reason"))}
        if result.get("status") == "ok":
            split = expected_splits[phase]
            prediction = load_predictions(directory/"predictions.npz", split.valid, len(test))
            if result.get("config_key", key) != key:
                raise ValueError("Ключ результата не совпадает с конфигурацией")
            if result.get("n_train") != len(split.train):
                raise ValueError("Размер обучения отличается от разбиения")
            if phase != "search" and result.get("best_iterations") != request["fixed_iterations"]:
                raise ValueError("Подтверждение/refit выполнили другое число итераций")
            if phase != "search" and result.get("evaluation_used_for_fit", False):
                raise ValueError("Оценочные строки использованы для обучения")
            if len(split.valid):
                metrics = evaluate(task, y[split.valid], prediction)
                check_score(metrics[task.metric], result.get("score"), directory.name)
                item["score"] = metrics[task.metric]
            if phase == "confirm":
                if key in confirm_by_key:
                    raise ValueError("Конфигурация подтверждалась несколько раз")
                confirm_by_key[key] = {"result": result, "prediction": prediction, "directory": directory.name}
            elif phase == "search":
                search_by_directory[directory.name] = {"result": result, "prediction": prediction, "config_key": key}
        checked_experiments.append(item)
    confirm_items = [v for v in checked_experiments if v["phase"] == "confirm"]
    if not confirm_items:
        raise ValueError("Нет подтверждающих запусков")
    ci_cache = {}
    confirm_indices = expected_splits["confirm"].valid

    def verify_plans(label):
        records = summary[label+"_plans"]
        allowed = {plan_identity(p) for p in frozen[label+"_plans"]}
        checked = []
        for record in records:
            if plan_identity(record) not in allowed:
                raise ValueError("Смесь выбрана после подтверждения: "+label)
            if len(record["members"]) != len(record["weights"]) or not np.isclose(sum(record["weights"]), 1):
                raise ValueError("Неверные веса смеси")
            prediction = sum(w*confirm_by_key[k]["prediction"] for w, k in zip(record["weights"], record["members"]))
            saved = load_predictions(work/record["predictions_path"], confirm_indices)
            np.testing.assert_array_equal(prediction, saved)
            metrics = evaluate(task, y[confirm_indices], prediction)
            check_score(metrics[task.metric], record["confirmation_score"], label)
            checked.append({**record, "metrics": metrics,
                            "per_class": class_metrics(task, y[confirm_indices], prediction)})
        best = sorted(checked, key=lambda r: r["confirmation_score"], reverse=task.maximize)[0]
        if plan_identity(best) != plan_identity(summary[label]):
            raise ValueError("Выбор лучшего плана не воспроизводится: "+label)
        check_score(best["confirmation_score"], summary[label]["confirmation_score"], label+" best")
        return best, checked

    baseline, baseline_plans = verify_plans("baseline")
    integrated, integrated_plans = verify_plans("integrated")
    reference = load_predictions(work/baseline["predictions_path"], confirm_indices)
    sign, base_score = (1 if task.maximize else -1), baseline["confirmation_score"]

    def comparison(prediction, score):
        delta = sign*(score-base_score)
        digest = hashlib.sha256(prediction.tobytes()).hexdigest()
        if digest not in ci_cache:
            ci_cache[digest] = ([0., 0.] if np.array_equal(prediction, reference) else paired_interval(
                task, y[confirm_indices], reference, prediction, groups[confirm_indices], repeats=400))
        return {"absolute_improvement_vs_baseline": delta,
                "relative_improvement_percent": delta/abs(base_score)*100 if base_score else None,
                "improvement_percentage_points": delta*100 if task.classification else None,
                "improvement_ci95_conditional": ci_cache[digest]}

    for label, record in (("integrated", integrated), ("selected", summary["selected"])):
        prediction = load_predictions(work/record["predictions_path"], confirm_indices)
        check_score(evaluate(task, y[confirm_indices], prediction)[task.metric], record["confirmation_score"], label)
        record.update(comparison(prediction, record["confirmation_score"]))
    expected_selected = integrated if better(task, integrated["confirmation_score"], base_score, .001) else baseline
    if plan_identity(expected_selected) != plan_identity(summary["selected"]):
        raise ValueError("Резервный выбор между новой ветвью и контролем не воспроизводится")
    methods = {}
    for name, original in summary["methods"].items():
        record = dict(original)
        if record.get("status") != "ok":
            methods[name] = record
            continue
        source_dir, confirm_dir = record["search_directory"], record["confirmation_directory"]
        source = search_by_directory[source_dir]
        key = source["config_key"]
        if frozen["methods"].get(name) != key or confirm_by_key[key]["directory"] != confirm_dir:
            raise ValueError("Метод не соответствует зафиксированному кандидату: "+name)
        search_metrics = evaluate(task, y[expected_splits["search"].valid], source["prediction"])
        check_score(search_metrics[task.metric], record["search_score"], name+" search")
        prediction = confirm_by_key[key]["prediction"]
        metrics = evaluate(task, y[confirm_indices], prediction)
        check_score(metrics[task.metric], record["confirmation_score"], name+" confirm")
        record.update(config_key=key, metrics=metrics, per_class=class_metrics(task, y[confirm_indices], prediction),
                      **comparison(prediction, metrics[task.metric]))
        methods[name] = record

    shared = read_json(work/"shared_llm_actions.json")
    if (summary["llm"]["calls"], summary["llm"]["tokens"]) != (shared["calls"], shared["tokens"]):
        raise ValueError("Учёт общего LLM-бюджета не совпадает между журналами")
    experiment_by_directory = {r["directory"]: r for r in checked_experiments}
    schedulers, pairs = {}, []
    expected_attempts = registered["scheduler_fit_attempts_per_policy_seed"]
    for seed in registered["scheduler_seeds"]:
        seed_records = {}
        for policy in registered["scheduler_policies"]:
            label = f"{policy}_{seed}"
            tree = read_json(work/(label+"_tree.json"))
            claimed = summary["schedulers"][label]
            if tree["policy"] != policy or tree["seed"] != seed or tree.get("pending_node_id") is not None:
                raise ValueError("Незавершённый либо неверный журнал планировщика: "+label)
            if tree["actions"] != shared["effective_actions"]:
                raise ValueError("Планировщики получили разные пулы действий")
            if (claimed["shared_llm_calls"], claimed["shared_llm_tokens"]) != (shared["calls"], shared["tokens"]):
                raise ValueError("LLM-бюджет политики отличается от общего пула")
            if tree["attempts"] != len(tree["nodes"])-1 or tree["attempts"] != claimed["attempts"]:
                raise ValueError("Число попыток планировщика не воспроизводится")
            if tree.get("extra_llm_calls", 0) or tree.get("extra_llm_tokens", 0) or claimed.get("extra_llm_calls", 0):
                raise ValueError("Одна из политик получила дополнительные LLM-вызовы")
            successful_nodes = [tree["nodes"][0]]
            root_node = tree["nodes"][0]
            root_source = search_by_directory[basename(root_node["directory"])]
            if root_node["config_key"] != root_source["config_key"]:
                raise ValueError("Исходный узел относится к другой конфигурации")
            check_score(root_source["result"]["score"], root_node["score"], label+" shared root")
            steps, nonsequential = [], 0
            for node in tree["nodes"][1:]:
                parent = tree["nodes"][node["parent"]]
                if parent["id"] >= node["id"] or node["id"] not in parent["children"]:
                    raise ValueError("Неверная связь родителей и потомков")
                if node["stage"] == "improve" and parent["status"] != "ok":
                    raise ValueError("Улучшение развивается от неуспешного родителя")
                if policy == "linear" and node["stage"] == "improve" and parent["id"] != successful_nodes[-1]["id"]:
                    raise ValueError("Линейная политика вернулась к старой успешной ветви")
                changed = [k for k in node["config"] if k != "hypothesis" and node["config"][k] != parent["config"][k]]
                if changed != [node["action"]["field"]] or node["config"][changed[0]] != node["action"]["value"]:
                    raise ValueError("Мутация изменила несколько полей либо не соответствует действию")
                directory = basename(node["directory"])
                request = request_map[directory]
                audited_experiment = experiment_by_directory[directory]
                if node["status"] != audited_experiment["status"]:
                    raise ValueError("Статус узла отличается от фактического результата исполнителя")
                check_score(audited_experiment["wall_seconds"], node["wall_seconds"], label+" actual duration")
                if parse_research_spec(request["config"], task).key != node["config_key"]:
                    raise ValueError("Журнал планировщика указывает на другой эксперимент")
                if node["status"] == "ok":
                    source = search_by_directory[directory]
                    check_score(source["result"]["score"], node["score"], label+" node")
                nonsequential += node["stage"] == "improve" and parent["id"] != successful_nodes[-1]["id"]
                steps.append({"id": node["id"], "parent": node["parent"], "stage": node["stage"],
                    "action": node["action"], "status": node["status"], "score": node["score"],
                    "wall_seconds": node["wall_seconds"], "error_type": node.get("error_type"), "error": node.get("error"),
                    "config_key": node["config_key"], "directory": directory})
                if node["status"] == "ok":
                    successful_nodes.append(node)
            best = sorted(successful_nodes, key=lambda n: n["score"], reverse=task.maximize)[0]
            check_score(best["score"], claimed["best_search_score"], label+" best")
            if best["config_key"] != claimed["config_key"]:
                raise ValueError("Планировщик выбрал другой config_key")
            record = {**claimed, "actions": tree["actions"], "action_order": tree["action_order"],
                      "nodes": steps, "complete_fit_attempt_budget": tree["attempts"] == expected_attempts,
                      "successful_fits": len(successful_nodes)-1,
                      "skipped_before_fit": sum(n["status"] == "skipped" for n in steps),
                      "shared_root_config_key": root_node["config_key"],
                      "nonsequential_parent_selections": int(nonsequential),
                      "confirmation_score": methods[label].get("confirmation_score"),
                      "confirmation_improvement_vs_baseline": methods[label].get("absolute_improvement_vs_baseline"),
                      "failed_recomputed": sum(n["status"] != "ok" for n in steps)}
            if record["failed_recomputed"] != claimed["failed"]:
                raise ValueError("Число ошибок планировщика не воспроизводится")
            schedulers[label] = seed_records[policy] = record
        left, right = seed_records["linear"], seed_records["tree"]
        if left["actions"] != right["actions"] or left["action_order"] != right["action_order"]:
            raise ValueError("У парных политик отличаются действия либо начальный порядок")
        if left["shared_root_config_key"] != right["shared_root_config_key"]:
            raise ValueError("Парные политики начали с разных исходных моделей")
        for field in ("wall_budget_seconds", "per_trial_cap_seconds", "shared_llm_calls", "shared_llm_tokens"):
            if left[field] != right[field]:
                raise ValueError("У политик неравные лимиты: "+field)
        linear_key = methods[f"linear_{seed}"].get("config_key")
        tree_key = methods[f"tree_{seed}"].get("config_key")
        tree_vs_linear_ci = None
        if linear_key in confirm_by_key and tree_key in confirm_by_key:
            linear_prediction = confirm_by_key[linear_key]["prediction"]
            tree_prediction = confirm_by_key[tree_key]["prediction"]
            tree_vs_linear_ci = ([0., 0.] if np.array_equal(linear_prediction, tree_prediction)
                else paired_interval(task, y[confirm_indices], linear_prediction, tree_prediction,
                                     groups[confirm_indices], repeats=400))
        pairs.append({"seed": seed, "attempts_equal": left["attempts"] == right["attempts"],
            "full_attempt_budget_both": left["complete_fit_attempt_budget"] and right["complete_fit_attempt_budget"],
            "action_pool_and_order_equal": True, "limits_equal": True,
            "tree_search_improvement_vs_linear": sign*(right["best_search_score"]-left["best_search_score"]),
            "tree_confirmation_improvement_vs_linear": (sign*(right["confirmation_score"]-left["confirmation_score"])
                if left["confirmation_score"] is not None and right["confirmation_score"] is not None else None),
            "tree_confirmation_improvement_ci95_conditional": tree_vs_linear_ci,
            "linear_actual_seconds": left["fit_seconds"], "tree_actual_seconds": right["fit_seconds"]})

    manifest = read_json(case/"docker_objects.json")
    resource = read_json(case/"resource_audit.json")
    dependencies = read_json(case/"dependency_audit.json") if (case/"dependency_audit.json").exists() else None
    if not resource["cpu_total"] <= 8 or not resource["memory_bytes_total"] <= 32000000000:
        raise ValueError("Превышены ресурсы ТЗ")
    if not (resource.get("internal_network") is True or resource.get("network") == "none"):
        raise ValueError("Не подтверждена изоляция сети")
    requirements_sha = hashlib.sha256((case/"requirements.txt").read_bytes()).hexdigest()
    if requirements_sha != REQUIREMENTS_SHA256:
        raise ValueError("Исходный requirements.txt изменён")
    if dependencies is not None and (dependencies.get("mismatches") or dependencies.get("requirements_sha256") != requirements_sha):
        raise ValueError("Версии зависимостей не соответствуют ТЗ")
    if summary["elapsed_seconds"] > summary["time_limit_seconds"] or summary["time_limit_seconds"] > 10800:
        raise ValueError("Превышен лимит полного запуска")
    archive_audit = {"recorded_sha256": manifest["archive_sha256"], "verified_archive": None,
                     "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(case.glob("*.py"))}}
    for path in (case/"source.zip", case/"alfa_kaggle_research.zip", ROOT/"dist"/"alfa_kaggle_research.zip"):
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == manifest["archive_sha256"]:
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None or any("/" in n or "\\" in n for n in archive.namelist()):
                    raise ValueError("Нарушен плоский формат либо целостность ZIP")
                for name in archive.namelist():
                    if archive.read(name) != (case/name).read_bytes():
                        raise ValueError("Исходник кейса отличается от исследовательского ZIP: "+name)
            archive_audit["verified_archive"] = str(path)
            break
    full_items = [r for r in checked_experiments if r["phase"] == "full"]
    if summary.get("final_refit_completed") and (len(full_items) != len(summary["selected"]["members"])
            or any(r["status"] != "ok" for r in full_items)
            or sorted(r["config_key"] for r in full_items) != sorted(summary["selected"]["members"])):
        raise ValueError("Заявленный полный refit не подтверждён журналами")
    try:
        relative_directory = str(case.relative_to(ROOT))
    except ValueError:
        relative_directory = str(case)
    return {"task": task.problem_type, "metric": task.metric, "directory": relative_directory,
        "summary_directory": work.name, "validation_note": summary.get("validation_note"),
        "bootstrap_note": "Парный пуассоновский bootstrap целых групп, 400 повторов; условный интервал не учитывает множественный выбор и будущий временной сдвиг.",
        "baseline": baseline, "integrated": integrated, "selected": summary["selected"],
        "baseline_plans": baseline_plans, "integrated_plans": integrated_plans, "methods": methods,
        "schedulers": schedulers, "scheduler_pairs": pairs, "shared_llm": shared,
        "elapsed_seconds": summary["elapsed_seconds"], "time_limit_seconds": summary["time_limit_seconds"],
        "peak_worker_rss_mb": max((r["peak_memory_mb"] for r in checked_experiments), default=0.),
        "final_refit_completed": summary["final_refit_completed"], "experiments": checked_experiments,
        "experiment_count": len(checked_experiments), "checks": checks,
        "resource_audit": resource, "dependency_audit": dependencies,
        "requirements_sha256": requirements_sha, "archive_audit": archive_audit,
        "python": summary.get("python"), "versions": summary.get("versions")}


def main():
    parser = argparse.ArgumentParser(description="Независимый пересчёт новых Kaggle/AIDE-экспериментов")
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--output", default=str(ROOT/"docs"/"KAGGLE_RESULTS.json"))
    args = parser.parse_args()
    results = [analyze(case) for case in args.cases]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(native(results), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for result in results:
        print(f'{result["task"]}: сильный контроль {result["baseline"]["confirmation_score"]:.6f}; '
              f'новый отбор {result["integrated"]["confirmation_score"]:.6f}; '
              f'проверено {result["experiment_count"]} запусков')
    print("Проверенный отчёт: "+str(output))


if __name__ == "__main__":
    main()
