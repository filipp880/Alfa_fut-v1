"""Проверяемый отчёт по предсказаниям, а не переписанным вручную метрикам."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
from contracts import native
from data_io import encode_target, feature_groups
from metrics import evaluate
from task_reader import load_task


def load_predictions(path):
    with np.load(path, allow_pickle=False) as values:
        return values["valid"], values["valid_indices"]


def paired_interval(task, y, reference, candidate, groups, repeats=400):
    """Парный пуассоновский bootstrap целых групп, условный на этой выборке."""
    if task.metric not in {"macro_f1", "rmse"}:
        return None
    _, inverse = np.unique(groups, return_inverse=True)
    count = inverse.max()+1
    if task.classification:
        classes = len(task.classes)
        shape = (count, classes*classes)
        table_reference = np.zeros(shape)
        table_candidate = np.zeros(shape)
        np.add.at(table_reference, (inverse, y*classes+reference.argmax(axis=1)), 1)
        np.add.at(table_candidate, (inverse, y*classes+candidate.argmax(axis=1)), 1)

        def score(table, weights):
            confusion = (weights@table).reshape(classes, classes)
            denominator = confusion.sum(axis=0)+confusion.sum(axis=1)
            return np.divide(2*np.diag(confusion), denominator,
                             out=np.zeros(classes), where=denominator > 0).mean()
    else:
        group_counts = np.bincount(inverse, minlength=count)
        table_reference = np.bincount(inverse, weights=(reference-y)**2, minlength=count)
        table_candidate = np.bincount(inverse, weights=(candidate-y)**2, minlength=count)

        def score(table, weights):
            return np.sqrt((weights@table)/max(weights@group_counts, 1))

    rng = np.random.default_rng(20261025)
    deltas = []
    for _ in range(repeats):
        weights = rng.poisson(1., count)
        difference = score(table_candidate, weights)-score(table_reference, weights)
        deltas.append(difference if task.maximize else -difference)
    return np.quantile(deltas, [.025, .975]).tolist()


def analyze(case):
    case = Path(case).resolve()
    paths = sorted(case.glob(".unconventional/*/summary.json"))
    summary_path = paths[-1]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] != "completed":
        raise ValueError("Исследование не завершено")
    work = summary_path.parent
    task, train, _, _, _ = load_task(case/"task.md")
    y = encode_target(task, train)
    groups = feature_groups(task, train)
    control = summary["methods"]["control"]
    reference, indices = load_predictions(work/control["confirmation_directory"]/"predictions.npz")
    checked = evaluate(task, y[indices], reference)[task.metric]
    if not np.isclose(checked, control["confirmation_score"], atol=1e-10):
        raise ValueError("Метрика контроля не воспроизводится по предсказаниям")
    frozen = work/"frozen_before_confirmation.json"
    confirm_requests = [p for p in work.glob("*confirm*/request.json")]
    if not confirm_requests or not all(frozen.stat().st_mtime <= p.stat().st_mtime for p in confirm_requests):
        raise ValueError("Нет подтверждения фиксации конфигураций до подтверждения")
    result = {"task": task.problem_type, "metric": task.metric,
              "directory": str(case.relative_to(ROOT)), "control_score": checked,
              "elapsed_seconds": summary["elapsed_seconds"], "winner": summary["winner"],
              "winner_alpha": summary["winner_alpha"], "peak_worker_rss_mb": summary["peak_worker_rss_mb"],
              "final_refit_completed": summary["final_refit_completed"], "methods": {},
              "requirements_audit": json.loads((case/"dependency_audit.json").read_text(encoding="utf-8")),
              "resource_audit": json.loads((case/"resource_audit.json").read_text(encoding="utf-8")),
              "archive_sha256": json.loads((case/"docker_objects.json").read_text(encoding="utf-8"))["archive_sha256"]}
    if (work/"kernel_ablation.json").exists():
        result["kernel_ablation"] = json.loads((work/"kernel_ablation.json").read_text(encoding="utf-8"))
    for name in ("extra_trees", "kernel", "retrieval"):
        record = dict(summary["methods"][name])
        directory = work/record["confirmation_directory"]
        pure, pure_indices = load_predictions(directory/"predictions.npz")
        blend, blend_indices = load_predictions(directory/"blend_predictions.npz")
        if not np.array_equal(indices, pure_indices) or not np.array_equal(indices, blend_indices):
            raise ValueError("Разные строки оценки")
        for label, prediction in (("confirmation_score", pure), ("confirmation_blend_score", blend)):
            actual = evaluate(task, y[indices], prediction)[task.metric]
            if not np.isclose(actual, record[label], atol=1e-10):
                raise ValueError("Метрика альтернативы не воспроизводится")
        sign = 1 if task.maximize else -1
        record["pure_improvement"] = sign*(record["confirmation_score"]-checked)
        record["pure_improvement_ci95_conditional"] = paired_interval(task, y[indices], reference, pure, groups[indices])
        record["blend_improvement"] = sign*(record["confirmation_blend_score"]-checked)
        record["blend_improvement_ci95_conditional"] = paired_interval(task, y[indices], reference, blend, groups[indices])
        if not task.classification:
            record["residual_correlation"] = float(np.corrcoef(reference-y[indices], pure-y[indices])[0, 1])
        result["methods"][name] = record
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--output", default=str(ROOT/"docs"/"UNCONVENTIONAL_RESULTS.json"))
    args = parser.parse_args()
    results = [analyze(path) for path in args.cases]
    output = Path(args.output)
    output.write_text(json.dumps(native(results), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    for result in results:
        print(f'{result["task"]}: контроль {result["control_score"]:.6f}; победитель {result["winner"]}')
        for name, values in result["methods"].items():
            print(f'{name}: отдельно {values["confirmation_score"]:.6f}, комбинация '
                  f'{values["confirmation_blend_score"]:.6f}, доля {values["alpha"]}, '
                  f'интервал улучшения {values["blend_improvement_ci95_conditional"]}')


if __name__ == "__main__":
    main()
