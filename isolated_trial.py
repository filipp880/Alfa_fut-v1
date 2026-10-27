"""Один вариант и одна фаза: замороженный контракт, маскированные метки, без сети.

Этот файл запускается только внутри контейнера, подготовленного isolated_trials.
Истинные исключённые метки остаются у внешнего проверяющего на host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

PHASES = ("search_early", "search_late", "search_main", "search_other", "confirm", "full")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _check_inputs(root, phase):
    """Проверить неизменяемые файлы и действительную маску до первого fit."""
    manifest = _json(root / "manifest.json")
    for name, record in manifest["immutable_files"].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or _digest(path) != record["sha256"]:
            raise ValueError("Изменён зарегистрированный вход: " + name)
    audit = _json(root / "phases" / phase / "masking_audit.json")
    if _digest(root / "train.csv") != audit["masked_train_sha256"]:
        raise ValueError("Хеш маскированного train не совпал")
    attestation = _json(root / "phases" / phase / "runtime_isolation_attestation.json")
    if not attestation.get("attested_before_fit") or attestation.get("status") != "verified":
        raise ValueError("Host не подтвердил действительную изоляцию")
    old_sources = list(Path("/agent").glob("*.py")) if Path("/agent").exists() else []
    if old_sources:
        raise ValueError("Старые исходники image доступны в /agent")
    permitted_root = set(manifest["immutable_files"]) | {"train.csv", "manifest.json"}
    permitted_dirs = {"splits", "phases"}
    for item in root.iterdir():
        if item.is_symlink() or (item.is_file() and item.name not in permitted_root):
            raise ValueError("Незарегистрированный файл в case: " + item.name)
        if item.is_dir() and item.name not in permitted_dirs:
            raise ValueError("Незарегистрированный каталог в case: " + item.name)
    for item in root.rglob("*"):
        if item.is_symlink():
            raise ValueError("Symlink внутри case запрещён")
        if item.is_file() and item.is_relative_to(root / "splits") and item.relative_to(root).as_posix() not in permitted_root:
            raise ValueError("Посторонний файл в frozen splits")
    registered_phases = set(_json(root / "split_manifest.json")["splits"])
    for item in (root / "phases").iterdir():
        if not item.is_dir() or item.name not in registered_phases:
            raise ValueError("Посторонний case внутри phases")
    return manifest, audit, attestation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--time-limit", type=float, default=10800)
    parser.add_argument("--trial-seconds", type=float, default=600)
    parser.add_argument("--fixed-iterations", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    if not 1 <= args.time_limit <= 10800 or not 0 < args.trial_seconds <= 10800:
        parser.error("Лимиты фазы и обучения должны быть в диапазоне 1..10800 секунд")
    if args.phase in {"confirm", "full"} and (args.fixed_iterations is None or args.fixed_iterations <= 0):
        parser.error("Для confirm/full нужен явный frozen fixed_iterations")
    if args.seed is not None and not 0 <= args.seed < 2**31:
        parser.error("Некорректный seed")

    from experiment_store import atomic_json
    from runtime import Budget, Runner
    import psutil

    directory = ROOT / "phases" / args.phase
    directory.mkdir(parents=True, exist_ok=True)
    summary = {"status": "initializing", "phase": args.phase, "network": "none", "llm_calls": 0,
               "label_blinded": args.phase == "confirm", "score": None,
               "time_limit_seconds": args.time_limit, "trial_seconds": args.trial_seconds}
    budget = Budget(STARTED, args.time_limit, 8, 26000)

    def save(status=None):
        if status:
            summary["status"] = status
        summary["elapsed_seconds"] = time.monotonic() - STARTED
        atomic_json(directory / "summary.json", summary)

    def stop():
        try:
            for child in reversed(psutil.Process().children(recursive=True)):
                try:
                    child.kill()
                except psutil.Error:
                    pass
            save("deadline")
        finally:
            os._exit(124)

    timer = threading.Timer(max(.1, budget.remaining() - .25), stop)
    timer.daemon = True
    timer.start()
    try:
        import numpy as np
        from contracts import TaskSpec, parse_experiment
        from data_io import encode_target, profile_data
        from task_reader import read_csv
        from validation import Split

        manifest, mask, attestation = _check_inputs(ROOT, args.phase)
        task = TaskSpec.from_dict(_json(ROOT / "task_contract.json"))
        for name in ("task_path", "train_path", "test_path", "sample_path", "dictionary_path"):
            value = getattr(task, name)
            if value is not None and Path(value).resolve().parent != ROOT:
                raise ValueError("Путь контракта покидает собственный case")
        train = read_csv(task.train_path, task.id_columns + task.categorical_columns)
        test = read_csv(task.test_path, task.id_columns + task.categorical_columns)
        dictionary = read_csv(task.dictionary_path).fillna("")
        y = encode_target(task, train)
        with np.load(ROOT / "splits" / (args.phase + ".npz"), allow_pickle=False) as values:
            tr, va = values["train"].copy(), values["valid"].copy()
        if any(a.ndim != 1 or a.dtype.kind not in "iu" or len(np.unique(a)) != len(a) for a in (tr, va)):
            raise ValueError("Индексы должны быть уникальными целыми одномерными массивами")
        if not len(tr) or np.intersect1d(tr, va).size or min(tr.min(), va.min() if len(va) else 0) < 0:
            raise ValueError("Некорректное разбиение")
        if max(tr.max(), va.max() if len(va) else 0) >= len(train):
            raise ValueError("Индекс вне train")
        if args.phase != "full":
            Split(args.phase, tr, va, "Зарегистрировано на host до маскирования").check(len(y), y, task.classification)
        elif len(va) or not np.array_equal(np.sort(tr), np.arange(len(train))):
            raise ValueError("Полный refit обязан покрывать весь train без validation")
        allowed = np.union1d(tr, va) if args.phase.startswith("search") else tr
        allowed_hash = hashlib.sha256(np.asarray(allowed, dtype="<i8").tobytes()).hexdigest()
        if allowed_hash != mask["allowed_label_indices_sha256"]:
            raise ValueError("Список разрешённых меток не совпал с host")

        profile, _, _ = profile_data(task, train.iloc[tr], test, dictionary)
        profile["time_column"] = _json(ROOT / "split_manifest.json")["time_column"]
        config_data = _json(ROOT / "config.json")
        if args.seed is not None:
            config_data["seed"] = args.seed
            for name in ("base_config", "history_config", "fallback_config"):
                if isinstance(config_data.get(name), dict):
                    config_data[name] = {**config_data[name], "seed": args.seed}
        config = parse_experiment(config_data, task)
        summary.update(config=config.to_dict(), config_key=config.key, source_manifest_sha256=_digest(ROOT / "manifest.json"),
                       masked_train_sha256=mask["masked_train_sha256"], allowed_label_rows=len(allowed),
                       n_train=len(tr), n_valid=len(va), profile=profile, fixed_iterations=args.fixed_iterations,
                       runtime_isolation_attestation=attestation, score=None)
        np.savez_compressed(directory / "split.npz", train=tr, valid=va)
        atomic_json(directory / "profile.json", profile)
        atomic_json(directory / "frozen_request.json", {"phase": args.phase, "config": config.to_dict(),
                    "fixed_iterations": args.fixed_iterations, "refit": args.phase in {"confirm", "full"},
                    "split_sha256": _digest(directory / "split.npz"), "masking_audit": mask})
        save("training")
        runner = Runner(directory, task, profile, budget)
        result = runner.run(config, directory / "split.npz", args.phase,
                            min(args.trial_seconds, budget.remaining()),
                            refit=args.phase in {"confirm", "full"}, fixed_iterations=args.fixed_iterations)
        work = Path(result["directory"])
        if args.phase == "confirm":
            # Worker оценивает фиктивные метки; эти числа не являются результатами.
            result["score"], result["metrics"] = None, {}
            for name in ("result.json", "controller_result.json"):
                if (work / name).exists():
                    record = _json(work / name)
                    record["score"], record["metrics"], record["label_blinded"] = None, {}, True
                    atomic_json(work / name, record)
        for name in ("request.json", "predictions.npz"):
            if (work / name).is_file():
                shutil.copyfile(work / name, directory / name)
        atomic_json(directory / "result.json", result)
        summary.update(worker_result=result, score=result.get("score"), metrics=result.get("metrics", {}),
                       best_iterations=result.get("best_iterations"),
                       actual_fit_count=result.get("component_fits", result.get("actual_fit_count", result.get("fit_count"))),
                       fit_count_details={k: v for k, v in result.items() if "fit_count" in k or k in {"component_fits", "component_iterations"}})
        if result.get("status") != "ok":
            save(result.get("status", "failed"))
            return 1
        with np.load(directory / "predictions.npz", allow_pickle=False) as predictions:
            if not np.array_equal(predictions["valid_indices"], va):
                raise ValueError("Worker изменил порядок validation indices")
            if len(predictions["valid"]) != len(va) or len(predictions["test"]) != len(test):
                raise ValueError("Worker вернул неверное число предсказаний")
            if not np.isfinite(predictions["valid"]).all() or not np.isfinite(predictions["test"]).all():
                raise ValueError("Неконечные предсказания")
        summary["prediction_sha256"] = _digest(directory / "predictions.npz")
        save("completed")
        return 0
    except Exception as error:
        summary["error_type"], summary["error"] = type(error).__name__, str(error)[-1800:]
        traceback.print_exc()
        save("failed")
        return 1
    finally:
        timer.cancel()


if __name__ == "__main__":
    raise SystemExit(main())
