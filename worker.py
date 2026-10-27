"""Точка входа контролируемого эксперимента; произвольный код не принимается."""
from __future__ import annotations
import json
from pathlib import Path
import pickle
import sys
import time
import traceback


def main(request_path: str) -> int:
    request_file = Path(request_path).resolve()
    output = request_file.parent
    started = time.monotonic()
    try:
        request = json.loads(request_file.read_text(encoding="utf-8"))
        if request.get("config", {}).get("model") == "proposal":
            from proposal_worker import main as proposal_main
            return proposal_main(request_path)
        if request.get("config", {}).get("model") in {"extra_trees", "kernel", "retrieval"}:
            from novel_worker import main as alternative_main
            return alternative_main(request_path)
        import numpy as np
        from threadpoolctl import threadpool_limits
        from contracts import ExperimentSpec, TaskSpec, parse_experiment
        from task_reader import read_csv
        from data_io import encode_target
        from models import train_model
        from metrics import evaluate
        from experiment_store import atomic_json
        request = json.loads(request_file.read_text(encoding="utf-8"))
        task = TaskSpec.from_dict(request["task"])
        config = parse_experiment(request["config"], task)
        train = read_csv(task.train_path, task.id_columns + task.categorical_columns)
        test = read_csv(task.test_path, task.id_columns + task.categorical_columns)
        y = encode_target(task, train)
        with np.load(request["split_path"], allow_pickle=False) as split:
            train_indices, valid_indices = split["train"], split["valid"]
        if set(train_indices.tolist()) & set(valid_indices.tolist()):
            raise ValueError("Пересечение индексов обучения и оценки")
        with threadpool_limits(limits=1, user_api="blas"):
            if config.model == "history_router":
                from history_router import train_router
                trainer = train_router
            else:
                trainer = train_model
            bundle, valid_pred, test_pred, details = trainer(
                task, config, request["profile"], train, test, y, train_indices, valid_indices,
                int(request["threads"]), bool(request.get("refit")), request.get("fixed_iterations"),
                max(.1, request.get("fit_time_limit_seconds", 3600)-(time.monotonic()-started)))
        metrics = evaluate(task, y[valid_indices], valid_pred) if len(valid_indices) else {}
        if len(test_pred) != len(test) or not np.isfinite(test_pred).all():
            raise ValueError("Некорректные предсказания теста; эксперимент отклонён")
        np.savez_compressed(output / "predictions.npz", valid=valid_pred, test=test_pred,
                            valid_indices=valid_indices)
        with (output / "model.pkl").open("wb") as stream:
            pickle.dump(bundle, stream, protocol=4)
        result = {"status": "ok", "config": config.to_dict(), "config_key": config.key,
                  "metrics": metrics, "score": metrics.get(task.metric),
                  "elapsed_seconds": time.monotonic()-started, **details}
        atomic_json(output / "result.json", result)
        return 0
    except Exception as error:
        from experiment_store import atomic_json
        atomic_json(output / "result.json", {"status": "failed", "error_type": type(error).__name__,
                     "error": str(error)[-1800:], "elapsed_seconds": time.monotonic()-started})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
