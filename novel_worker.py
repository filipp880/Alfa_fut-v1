"""Изолированное обучение альтернатив; тот же протокол, что у основного worker."""
from pathlib import Path
import json
import sys
import time
import traceback


def main(request_path):
    path = Path(request_path).resolve()
    request = json.loads(path.read_text(encoding="utf-8"))
    if request["config"]["model"] in {"catboost", "lightgbm", "linear"}:
        from worker import main as standard_main
        return standard_main(request_path)
    started = time.monotonic()
    try:
        import joblib
        import numpy as np
        from threadpoolctl import threadpool_limits
        from contracts import TaskSpec
        from data_io import encode_target
        from experiment_store import atomic_json
        from metrics import evaluate
        from novel_models import AlternativeModel, NovelSpec
        from task_reader import read_csv
        task = TaskSpec.from_dict(request["task"])
        config = NovelSpec.from_dict(request["config"], task)
        train = read_csv(task.train_path, task.id_columns+task.categorical_columns)
        test = read_csv(task.test_path, task.id_columns+task.categorical_columns)
        y = encode_target(task, train)
        with np.load(request["split_path"], allow_pickle=False) as data:
            tr, va = data["train"], data["valid"]
        if not len(tr) or np.intersect1d(tr, va).size:
            raise ValueError("Некорректное разбиение альтернативного эксперимента")
        if any(index < 0 or index >= len(train) for index in np.r_[tr, va]):
            raise ValueError("Индексы вне таблицы")
        with threadpool_limits(limits=request["threads"]):
            model = AlternativeModel(task, config, request["threads"]).fit(train.iloc[tr], y[tr])
            trained = time.monotonic()
            valid, predicted = model.predict(train.iloc[va]), model.predict(test)
        metrics = evaluate(task, y[va], valid) if len(va) else {}
        if not np.isfinite(predicted).all() or not np.isfinite(valid).all():
            raise ValueError("Модель вернула неконечные предсказания")
        joblib.dump(model, path.parent/"model.joblib", compress=0)
        np.savez_compressed(path.parent/"predictions.npz", valid=valid, test=predicted, valid_indices=va)
        atomic_json(path.parent/"result.json", {"status": "ok", "score": metrics.get(task.metric),
            "metrics": metrics, "best_iterations": 1, "n_train": len(tr),
            "feature_count": model.feature_count, "training_seconds": trained-started,
            "prediction_seconds": time.monotonic()-trained, "elapsed_seconds": time.monotonic()-started})
        return 0
    except Exception as error:
        from experiment_store import atomic_json
        atomic_json(path.parent/"result.json", {"status": "failed", "error_type": type(error).__name__,
                    "error": str(error)[-1800:], "elapsed_seconds": time.monotonic()-started})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
