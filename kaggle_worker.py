"""Изолированный исполнитель новых Kaggle-гипотез с общим Runner/watchdog."""
from dataclasses import replace
from pathlib import Path
import json
import pickle
import sys
import time
import traceback


def main(request_path):
    path = Path(request_path).resolve()
    request = json.loads(path.read_text(encoding="utf-8"))
    if request["config"]["model"] not in {"augmentation", "xgboost"}:
        from novel_worker import main as standard
        return standard(request_path)
    started = time.monotonic()
    try:
        import numpy as np
        import pandas as pd
        from threadpoolctl import threadpool_limits
        from contracts import TaskSpec, parse_experiment
        from data_io import encode_target
        from experiment_store import atomic_json
        from kaggle_specs import parse_research_spec
        from metrics import evaluate
        from models import train_model
        from task_reader import read_csv
        task = TaskSpec.from_dict(request["task"])
        config = parse_research_spec(request["config"], task)
        train = read_csv(task.train_path, task.id_columns+task.categorical_columns)
        test = read_csv(task.test_path, task.id_columns+task.categorical_columns)
        y = encode_target(task, train)
        with np.load(request["split_path"], allow_pickle=False) as values:
            tr, va = values["train"], values["valid"]
        if not len(tr) or np.intersect1d(tr, va).size or any(i < 0 or i >= len(train) for i in np.r_[tr, va]):
            raise ValueError("Некорректное разбиение")
        threads = int(request["threads"])
        diagnostics, transforms = {}, []
        with threadpool_limits(limits=1, user_api="blas"):
            if config.model == "augmentation":
                xtr, xva, xte = (frame[task.feature_columns].copy() for frame in (train.iloc[tr], train.iloc[va], test))
                forbidden = task.id_columns+[task.target]
                if config.feature_schema:
                    from kaggle_features import FoldFeatureAugmenter
                    features = FoldFeatureAugmenter(config.feature_schema, forbidden_columns=forbidden,
                                                   seed=config.seed, threads=threads)
                    xtr = features.fit_transform(xtr)
                    xva, xte = features.transform(xva), features.transform(xte)
                    diagnostics["features"] = features.diagnostics_
                    transforms.append(features)
                if config.history_schema:
                    from historical_features import HistoricalFeatures, HistorySchema
                    from historical_features import HISTORY_COLUMNS
                    schema = dict(config.history_schema)
                    schema["forbidden_columns"] = forbidden
                    history = HistoricalFeatures(HistorySchema(**schema))
                    history_train = history.fit_transform(train.iloc[tr][task.feature_columns], y[tr])
                    diagnostics["history_fit"] = history.diagnostics_
                    history_valid = history.transform(train.iloc[va][task.feature_columns])
                    diagnostics["history_valid"] = history.last_diagnostics_
                    history_test = history.transform(test[task.feature_columns])
                    diagnostics["history_test"] = history.last_diagnostics_
                    for frame, values in ((xtr, history_train),(xva, history_valid),(xte, history_test)):
                        for column in HISTORY_COLUMNS:
                            frame[column] = values[column].to_numpy()
                    transforms.append(history)
                augmented = train.copy()
                for col in xtr:
                    if col not in augmented:
                        augmented[col] = np.nan
                    augmented.loc[train.index[tr], col] = xtr[col].to_numpy()
                    if len(va):
                        augmented.loc[train.index[va], col] = xva[col].to_numpy()
                augmented_test = test.copy()
                for col in xte:
                    augmented_test[col] = xte[col].to_numpy()
                augmented_task = replace(task, feature_columns=list(xtr.columns))
                base = parse_experiment(config.base_config, augmented_task)
                if base.model == "extra_trees":
                    from novel_models import AlternativeModel
                    model = AlternativeModel(augmented_task, base, threads).fit(augmented.iloc[tr], y[tr])
                    valid, predicted = model.predict(augmented.iloc[va]), model.predict(augmented_test)
                    details = {"best_iterations": 1, "feature_count": model.feature_count,
                               "n_train": len(tr), "training_seconds": time.monotonic()-started,
                               "prediction_seconds": 0}
                    bundle = {"model": model}
                else:
                    bundle, valid, predicted, details = train_model(
                        augmented_task, base, request["profile"], augmented, augmented_test, y, tr, va,
                        threads, request.get("refit", False), request.get("fixed_iterations"),
                        max(.1, request.get("fit_time_limit_seconds", 3600)-(time.monotonic()-started)))
                bundle["augmenters"] = transforms
            else:
                from kaggle_xgb import train_xgboost
                bundle, valid, predicted, details = train_xgboost(
                    task, config, request["profile"], train, test, y, tr, va, threads,
                    request.get("refit", False), request.get("fixed_iterations"),
                    max(.1, request.get("fit_time_limit_seconds", 3600)-(time.monotonic()-started)))
        if len(predicted) != len(test) or not np.isfinite(predicted).all() or not np.isfinite(valid).all():
            raise ValueError("Некорректные предсказания")
        metrics = evaluate(task, y[va], valid) if len(va) else {}
        np.savez_compressed(path.parent/"predictions.npz", valid=valid, test=predicted, valid_indices=va)
        with (path.parent/"model.pkl").open("wb") as stream:
            pickle.dump(bundle, stream, protocol=4)
        atomic_json(path.parent/"result.json", {"status": "ok", "score": metrics.get(task.metric),
            "metrics": metrics, "diagnostics": diagnostics, **details, "elapsed_seconds": time.monotonic()-started})
        return 0
    except Exception as error:
        from experiment_store import atomic_json
        atomic_json(path.parent/"result.json", {"status": "failed", "error_type": type(error).__name__,
            "error": str(error)[-1800:], "elapsed_seconds": time.monotonic()-started})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
