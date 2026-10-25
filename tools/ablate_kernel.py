"""После основного сравнения: удалить только ядро, сохранив линейную геометрию.

Вспомогательный .py копируется в корень того же исследовательского кейса.
Не меняет конфигурации, основную таблицу результатов или submission.csv.
"""
from pathlib import Path
import json
import os
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"


def main():
    started = time.monotonic()
    timer = threading.Timer(300, lambda: os._exit(2))
    timer.daemon = True
    timer.start()
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression, Ridge
        from threadpoolctl import threadpool_limits
        from task_reader import load_task
        from data_io import encode_target
        from experiment_store import atomic_json
        from metrics import evaluate, normalized_probabilities
        from models import sample_weights
        from novel_models import MixedEncoder
        root = Path(sys.argv[1] if len(sys.argv) > 1 else "task.md").resolve().parent
        summary_path = sorted(root.glob(".unconventional/*/summary.json"))[-1]
        work = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["status"] != "completed":
            raise ValueError("Абляция допустима только после завершённого основного сравнения")
        record = summary["methods"]["kernel"]
        config, alpha = record["config"], record["alpha"]
        task, train, _, _, _ = load_task(root/"task.md")
        y = encode_target(task, train)
        controls = [r for r in summary["experiments"]
                    if r["phase"] == "control_search" and r["status"] == "ok"]
        search_control = sorted(controls, key=lambda r: r["score"], reverse=task.maximize)[0]
        result = {"exploratory_ablation_after_primary_comparison": True,
                  "hypothesis": "Даёт ли ядро полезный сигнал сверх линейной части?",
                  "alpha_frozen_from_primary_search": alpha,
                  "regularization_frozen_from_primary_search": config["regularization"], "phases": {}}
        for phase in ("search", "confirm"):
            with np.load(work/f"{phase}.npz", allow_pickle=False) as split:
                tr, va = split["train"], split["valid"]
            base_dir = Path(search_control["directory"]).name if phase == "search" else summary["methods"]["control"]["confirmation_directory"]
            with np.load(work/base_dir/"predictions.npz", allow_pickle=False) as values:
                base = values["valid"]
                if not np.array_equal(values["valid_indices"], va):
                    raise ValueError("Неверные строки абляции")
            with threadpool_limits(limits=7):
                encoder = MixedEncoder()
                x = encoder.fit_transform(task, train.iloc[tr])
                validation = encoder.transform(train.iloc[va])
                divisor = np.sqrt(max(1, x.shape[1]))
                x, validation = x/divisor, validation/divisor
                weights = sample_weights(y[tr], len(task.classes), config["class_weight_power"]) if task.classification else None
                if task.classification:
                    model = LogisticRegression(C=1/config["regularization"], max_iter=300,
                                               tol=1e-3, solver="lbfgs", random_state=config["seed"])
                    model.fit(x, y[tr], sample_weight=weights)
                    prediction = np.zeros((len(va), len(task.classes)))
                    prediction[:, model.classes_.astype(int)] = model.predict_proba(validation)
                    prediction = normalized_probabilities(prediction, len(task.classes))
                else:
                    model = Ridge(alpha=config["regularization"], solver="cholesky").fit(x, y[tr])
                    prediction = model.predict(validation)
                pure = evaluate(task, y[va], prediction)[task.metric]
                blended = evaluate(task, y[va], (1-alpha)*base+alpha*prediction)[task.metric]
            result["phases"][phase] = {"linear_only_score": pure, "linear_blend_score": blended,
                "full_kernel_score": record["search_score" if phase == "search" else "confirmation_score"],
                "full_kernel_blend_score": record["search_blend_score" if phase == "search" else "confirmation_blend_score"]}
        result["elapsed_seconds"] = time.monotonic()-started
        atomic_json(work/"kernel_ablation.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        timer.cancel()


if __name__ == "__main__":
    main()
