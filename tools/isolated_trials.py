"""Host-обвязка независимых офлайн-вариантов: один bind mount на один case.

create_trial не сохраняет исходный train с истинными исключёнными метками.
run_phase получает путь к оригиналу только на host и перед каждым запуском
пересобирает разрешённую маску. Пути оригинала не передаются контейнеру.
"""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ("task.md", "train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv")
PHASES = {"search_early", "search_late", "search_main", "search_other", "confirm", "full"}
MEMORY_BYTES = 29 * 1024**3


def _write_masked_csv(original, destination, target, allowed, reference_index):
    """Менять только raw target cells, сохраняя исходные X и точность десятичных."""
    with Path(original).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = list(reader)
    if header.count(target) != 1 or any(len(row) != len(header) for row in rows):
        raise ValueError("Неоднозначный таргет или повреждённый CSV")
    column = header.index(target)
    replacement = rows[int(reference_index)][column]
    allowed = set(int(index) for index in allowed)
    for index, row in enumerate(rows):
        if index not in allowed:
            row[column] = replacement
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    temporary.replace(destination)
    return len(rows) - len(allowed)


def _write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path):
    return {"sha256": _sha(path), "bytes": Path(path).stat().st_size}


def _verify_immutable(directory, manifest):
    for name, record in manifest["immutable_files"].items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or _sha(path) != record["sha256"]:
            raise ValueError("Изменён замороженный вход: " + name)


def _clean_generated_bytecode(directory, manifest):
    """Удалить только кеш собственных зарегистрированных модулей, не артефакты."""
    directory = Path(directory).resolve()
    cache = directory / "__pycache__"
    if not cache.exists(): return
    if cache.is_symlink() or cache.resolve().parent != directory or not cache.is_dir():
        raise ValueError("Небезопасный путь кеша Python")
    permitted = {Path(name).stem for name in manifest["source_files"] if name.endswith(".py")}
    entries = list(cache.iterdir())
    if any(p.is_symlink() or not p.is_file() or p.suffix != ".pyc" or p.name.split(".", 1)[0] not in permitted for p in entries):
        raise ValueError("Кеш содержит посторонний файл")
    for path in entries: path.unlink()
    cache.rmdir()


def _validate_config(value):
    """Конфиг содержит только параметры рецепта, без ссылок на чужие результаты."""
    if not isinstance(value, dict):
        raise ValueError("Конфигурация должна быть JSON-объектом")
    def visit(node):
        if isinstance(node, dict):
            for key, item in node.items():
                if not isinstance(key, str):
                    raise ValueError("Ключи конфигурации должны быть строками")
                lower = key.lower()
                if (lower in {"score", "scores", "metrics", "directory", "path", "predictions", "result_file"}
                        or lower.endswith(("_path", "_directory", "_score", "_scores", "_file"))):
                    raise ValueError("Ссылки на файлы/результаты запрещены в конфиге: " + key)
                visit(item)
        elif isinstance(node, (list, tuple)):
            for item in node:
                visit(item)
        elif isinstance(node, str):
            if (re.match(r"^[A-Za-z]:[\\/]", node) or node.startswith(("/", "\\"))
                    or re.search(r"(?:^|[\\/])\.\.(?:[\\/]|$)|://", node)
                    or re.search(r"(?:^|[\\/])(?:runs|\.ai_proposals|\.alfa_runs)(?:[\\/]|$)", node, re.I)):
                raise ValueError("Пути и URL запрещены в конфигурации")
    visit(value)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _load_host_modules():
    # Импорт общего доверенного кода нужен только на host при подготовке.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import numpy as np
    from data_io import encode_target, profile_data
    from task_reader import load_task, read_csv
    from validation import Split, make_splits
    return np, encode_target, profile_data, load_task, read_csv, Split, make_splits


def _phase_splits(task, y, groups, times):
    """Общие разбиения фиксируются до первого маскирования таргета."""
    np, _, _, _, _, Split, make_splits = _load_host_modules()
    standard = make_splits(task, y, groups, times)
    if len(standard) < 2:
        raise ValueError("Строгому стенду нужны отдельные search и confirm")
    search, confirm = standard[:2]
    if task.classification:
        from sklearn.model_selection import StratifiedGroupKFold
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
        parts = [va for _, va in cv.split(np.arange(len(y)), y, groups)]
        # Один зафиксированный splitter для обоих search и confirm.
        confirm = Split("confirm", np.setdiff1d(np.arange(len(y)), parts[1]), parts[1], "SGKF5: fold1 confirm")
        main = Split("search_main", np.setdiff1d(np.arange(len(y)), np.r_[parts[0], parts[1]]), parts[0], "SGKF5: fold0 search")
        other = Split("search_other", np.setdiff1d(np.arange(len(y)), np.r_[parts[1], parts[2]]), parts[2], "SGKF5: fold2 search")
        searches = [main, other]
    else:
        if times is None:
            raise ValueError("Регрессионному стенду нужна временная ось")
        levels = np.unique(times)
        if len(levels) < 4:
            raise ValueError("Для двух временных search и confirm нужны минимум 4 периода")
        if len(levels) <= 20:
            cut = levels[-3]
            early = Split("search_early", np.flatnonzero(times < cut), np.flatnonzero(times == cut), "Ранний временной search")
        else:
            earlier_levels = np.unique(times[search.train])
            cut = earlier_levels[max(1, int(len(earlier_levels) * 2 / 3))]
            early = Split("search_early", np.flatnonzero(times < cut),
                          search.train[times[search.train] >= cut], "Ранний блок внутри main-search train")
        late = Split("search_late", search.train, search.valid, search.reason)
        searches = [early, late]
    result = {split.name: split for split in searches + [confirm]}
    for name, split in result.items():
        split.check(len(y), y, task.classification)
        if len(np.unique(split.train)) != len(split.train) or len(np.unique(split.valid)) != len(split.valid):
            raise ValueError("Повторные индексы разбиения")
        if name.startswith("search") and np.intersect1d(np.r_[split.train, split.valid], confirm.valid).size:
            raise ValueError("Confirm пересекается с поисковой фазой")
        if task.classification and np.intersect1d(groups[split.train], groups[split.valid]).size:
            raise ValueError("Группа пересекает train и validation")
        if not task.classification and np.max(times[split.train]) >= np.min(times[split.valid]):
            raise ValueError("Обучение не предшествует временной проверке")
    if np.intersect1d(searches[0].valid, searches[1].valid).size:
        raise ValueError("Validation двух search пересекаются")
    result["full"] = Split("full", np.arange(len(y)), np.empty(0, dtype=np.int64), "Полный refit")
    return result


def create_trial(rootrun, label, task_source, spec_dict, source_modules):
    """Подготовить уникальный case; вернуть Path. Обучение/Docker не запускаются.

    source_modules — список доверенных .py/requirements*.txt, относительных к
    checkout либо абсолютных. В case копируются только их плоские имена.
    """
    if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", label) or ".." in label:
        raise ValueError("Небезопасное имя варианта")
    config = _validate_config(spec_dict)
    source = Path(task_source).resolve()
    task_file = source / "task.md" if source.is_dir() else source
    inputs = {name: task_file if name == "task.md" else task_file.parent / name for name in INPUTS}
    if any(not path.is_file() or path.is_symlink() for path in inputs.values()):
        raise ValueError("Нужны пять обычных входных файлов без symlink")
    modules = {}
    items = list(source_modules)
    if not any(Path(item).name == "isolated_trial.py" for item in items):
        items.append(ROOT / "isolated_trial.py")
    for item in items:
        path = Path(item)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_symlink():
            raise ValueError("Symlink исходников запрещён")
        path = path.resolve()
        if not path.is_file() or not (path.suffix == ".py" or re.fullmatch(r"requirements(?:-[A-Za-z0-9_-]+)?\.txt", path.name)):
            raise ValueError("Разрешены только trusted .py и requirements*.txt")
        if path.name in modules and modules[path.name] != path:
            raise ValueError("Коллизия плоских имён исходников: " + path.name)
        modules[path.name] = path
    np, encode_target, profile_data, load_task, _, _, _ = _load_host_modules()
    task, train, test, _, dictionary = load_task(task_file)
    # Все входы обязаны совпасть с теми пятью файлами, которые будут смонтированы.
    for field, expected in (("train_path", "train.csv"), ("test_path", "test.csv"),
                            ("sample_path", "sample_submission.csv"), ("dictionary_path", "data_dictionary.csv")):
        if Path(getattr(task, field)).resolve() != inputs[expected].resolve():
            raise ValueError("Task ссылается за пределы пяти зарегистрированных входов")
    y = encode_target(task, train)
    profile, groups, times = profile_data(task, train, test, dictionary)
    splits = _phase_splits(task, y, groups, times)
    destination = Path(rootrun).resolve() / label
    destination.mkdir(parents=True, exist_ok=False)
    for name, path in inputs.items():
        if name != "train.csv":
            shutil.copyfile(path, destination / name)
    for name, path in modules.items():
        shutil.copyfile(path, destination / name)
    _write_json(destination / "config.json", config)
    contract = task.to_dict()
    for field, name in (("task_path", "task.md"), ("train_path", "train.csv"), ("test_path", "test.csv"),
                        ("sample_path", "sample_submission.csv"), ("dictionary_path", "data_dictionary.csv")):
        contract[field] = "/task/" + name
    _write_json(destination / "task_contract.json", contract)
    (destination / "splits").mkdir()
    split_info = {}
    for name, split in splits.items():
        np.savez_compressed(destination / "splits" / (name + ".npz"), train=split.train, valid=split.valid)
        split_info[name] = {"train_rows": len(split.train), "valid_rows": len(split.valid), "reason": split.reason}
    _write_json(destination / "split_manifest.json", {"time_column": profile["time_column"], "splits": split_info,
                "seed": 42, "target_exported": False, "contract_classes_frozen": True})
    # В case ни на мгновение не попадает копия истинного train.
    _write_masked_csv(inputs["train.csv"], destination / "train.csv", task.target, (),
                      splits[next(iter(splits))].train[0])
    immutable = [name for name in INPUTS if name != "train.csv"] + list(modules) + ["config.json", "task_contract.json", "split_manifest.json"]
    immutable += ["splits/" + name + ".npz" for name in splits]
    manifest = {"schema_version": 1, "label": label, "created_utc": datetime.now(timezone.utc).isoformat(),
                "original_input_hashes": {name: _record(path) for name, path in inputs.items()},
                "immutable_files": {name: _record(destination / name) for name in immutable},
                "source_files": sorted(modules), "initial_masked_train_sha256": _sha(destination / "train.csv"),
                "original_train_mounted": False, "llm_calls": 0}
    _write_json(destination / "manifest.json", manifest)
    return destination


def _mask_phase(directory, phase, trusted_train_path):
    np, _, _, _, read_csv, _, _ = _load_host_modules()
    manifest = _read_json(directory / "manifest.json")
    original = Path(trusted_train_path).resolve()
    if original.is_relative_to(directory) or _sha(original) != manifest["original_input_hashes"]["train.csv"]["sha256"]:
        raise ValueError("Нужен неизменённый оригинальный train вне case")
    contract = _read_json(directory / "task_contract.json")
    with np.load(directory / "splits" / (phase + ".npz"), allow_pickle=False) as split:
        tr, va = split["train"], split["valid"]
    allowed = np.union1d(tr, va) if phase.startswith("search") else tr
    target = contract["target"]
    masked_rows = _write_masked_csv(original, directory / "train.csv", target, allowed, tr[0])
    audit = {"phase": phase, "original_train_sha256": _sha(original), "masked_train_sha256": _sha(directory / "train.csv"),
             "allowed_label_indices_sha256": hashlib.sha256(np.asarray(allowed, dtype="<i8").tobytes()).hexdigest(),
             "allowed_label_rows": len(allowed), "masked_label_rows": masked_rows,
             "raw_feature_values_preserved": True,
             "validation_labels_available": phase.startswith("search"), "label_blinded": phase == "confirm",
             "original_train_mounted": False}
    _write_json(directory / "phases" / phase / "masking_audit.json", audit)
    return audit


def _source_matches(actual, expected):
    actual = str(actual).replace("\\", "/").rstrip("/").lower()
    expected = str(expected).replace("\\", "/").rstrip("/").lower()
    accepted = {expected}
    if re.match(r"^[a-z]:/", expected):
        drive, tail = expected[0], expected[3:]
        accepted |= {prefix + drive + "/" + tail for prefix in ("/host_mnt/", "/run/desktop/mnt/host/", "/mnt/host/")}
    return actual in accepted


def _audit_container(info, directory, container, *, expected_cpus=8, expected_memory_bytes=MEMORY_BYTES):
    if not 1 <= expected_cpus <= 8 or not 0 < expected_memory_bytes <= 32000000000:
        raise ValueError("Некорректные ожидаемые ресурсы")
    host = info["HostConfig"]
    mounts = info.get("Mounts", [])
    binds = [mount for mount in mounts if mount["Type"] == "bind"]
    if (len(binds) != 1 or binds[0]["Destination"] != "/task"
            or not _source_matches(binds[0]["Source"], directory) or not binds[0].get("RW")):
        raise ValueError("Контейнер должен иметь ровно один bind: собственный case → /task")
    if any(mount["Type"] != "bind" and not (mount["Type"] == "tmpfs" and mount["Destination"] in {"/tmp", "/agent"}) for mount in mounts):
        raise ValueError("Обнаружен посторонний mount")
    tmpfs = host.get("Tmpfs") or {}
    if set(tmpfs) != {"/tmp", "/agent"} or "noexec" not in tmpfs["/agent"]:
        raise ValueError("Нужны только /tmp tmpfs и скрывающий старый код /agent noexec tmpfs")
    if (host.get("NetworkMode") != "none" or host.get("NanoCpus") != expected_cpus*1_000_000_000
            or host.get("Memory") != expected_memory_bytes or host.get("MemorySwap") != expected_memory_bytes
            or not host.get("ReadonlyRootfs") or host.get("Privileged")
            or "ALL" not in (host.get("CapDrop") or [])
            or not any(str(item).startswith("no-new-privileges") for item in host.get("SecurityOpt") or [])
            or host.get("PidMode") == "host" or host.get("IpcMode") == "host"
            or host.get("Devices") or host.get("DeviceRequests")):
        raise ValueError("Действительные ресурсы/изоляция Docker не совпадают с протоколом")
    environment = dict(item.split("=", 1) for item in info.get("Config", {}).get("Env", []) if "=" in item)
    for key, value in environment.items():
        if value and (re.search(r"(?:API_KEY|TOKEN|SECRET|PASSWORD|_PROXY)$", key, re.I)):
            raise ValueError("В image обнаружена запрещённая переменная окружения: " + key)
    return {"status": "verified", "attested_before_fit": True, "container": container, "image_id": info["Image"],
            "actual_bind_mounts": [{"source": mount["Source"], "destination": mount["Destination"], "rw": mount["RW"]} for mount in binds],
            "tmpfs": tmpfs, "network": host["NetworkMode"], "cpu_total": host["NanoCpus"] / 1e9,
            "memory_bytes_total": host["Memory"], "memory_swap_bytes": host["MemorySwap"],
            "readonly_root": host["ReadonlyRootfs"], "cap_drop": host["CapDrop"],
            "no_socket_or_home_mounts": True, "image_checkout_hidden": True, "llm_calls": 0}


def run_phase(directory, phase, time_limit=10800, trial_seconds=600, fixed_iterations=None, seed=None,
              *, trusted_train_path, image="alfa-research:local"):
    """Последовательно выполнить одну фазу; вернуть пути, summary и resource audit.

    Для confirm/full fixed_iterations обязательно берётся из main-search.
    trusted_train_path остаётся исключительно в host-процессе.
    """
    started = time.monotonic()
    deadline = started + float(time_limit)
    directory = Path(directory).resolve()
    if phase not in PHASES or not (directory / "splits" / (phase + ".npz")).is_file():
        raise ValueError("Фаза не зарегистрирована в этом case")
    if not 1 <= float(time_limit) <= 10800 or not 0 < float(trial_seconds) <= 10800:
        raise ValueError("Неверный лимит времени")
    if fixed_iterations is not None and (isinstance(fixed_iterations, bool) or not isinstance(fixed_iterations, int) or fixed_iterations <= 0):
        raise ValueError("fixed_iterations должен быть положительным целым")
    if phase in {"confirm", "full"} and fixed_iterations is None:
        raise ValueError("Confirm/full без frozen main-search count запрещены")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31):
        raise ValueError("Некорректный seed")
    phase_directory = directory / "phases" / phase
    phase_directory.mkdir(parents=True, exist_ok=False)
    manifest = _read_json(directory / "manifest.json")
    manifest_hash = _sha(directory / "manifest.json")
    _verify_immutable(directory, manifest)
    _clean_generated_bytecode(directory, manifest)
    masking = _mask_phase(directory, phase, trusted_train_path)
    container = "alfa-isolated-" + uuid.uuid4().hex[:16]
    log = phase_directory / "console.log"
    attestation = None
    created = False
    def docker(*arguments, **kwargs):
        return subprocess.run(["docker", *arguments], check=True, timeout=max(.1, deadline - time.monotonic()), **kwargs)
    try:
        image_id = docker("image", "inspect", image, "--format", "{{.Id}}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip()
        command = ["create", "--name", container, "--pull=never", "--network=none", "--cpus=8", "--memory=29g", "--memory-swap=29g",
                   "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--tmpfs", "/tmp:rw,size=2g",
                   "--tmpfs", "/agent:rw,noexec,size=1m", "--mount", f"type=bind,source={directory},target=/task", "--workdir", "/task"]
        for value in ("PYTHONDONTWRITEBYTECODE=1", "PYTHONIOENCODING=utf-8", "PYTHONUNBUFFERED=1", "HOME=/tmp/home",
                      "XDG_CACHE_HOME=/tmp/cache", "MPLCONFIGDIR=/tmp/matplotlib", "HTTP_PROXY=", "HTTPS_PROXY=", "ALL_PROXY=", "NO_PROXY="):
            command.extend(["--env", value])
        command += ["--entrypoint", "python3", image_id, "/task/isolated_trial.py", phase,
                    "--time-limit", str(max(1., float(time_limit) - 3)), "--trial-seconds", str(trial_seconds)]
        if fixed_iterations is not None:
            command += ["--fixed-iterations", str(fixed_iterations)]
        if seed is not None:
            command += ["--seed", str(seed)]
        docker(*command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        created = True
        inspected = json.loads(docker("inspect", container, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)[0]
        attestation = _audit_container(inspected, directory, container)
        attestation.update(manifest_sha256=_sha(directory / "manifest.json"), masking_audit=masking)
        _write_json(phase_directory / "runtime_isolation_attestation.json", attestation)
        _write_json(phase_directory / "host_request.json", {"phase": phase, "image_id": image_id, "container": container,
                    "time_limit_seconds": time_limit, "trial_seconds": trial_seconds,
                    "fixed_iterations": fixed_iterations, "seed": seed, "trusted_original_mounted": False})
        with log.open("w", encoding="utf-8") as stream:
            completed = docker("start", "--attach", container, stdout=stream, stderr=subprocess.STDOUT)
        state = json.loads(docker("inspect", container, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)[0]["State"]
        if state.get("ExitCode") != 0 or state.get("OOMKilled"):
            raise RuntimeError("Контейнер завершился неуспешно: " + str(state.get("ExitCode")))
        summary = _read_json(phase_directory / "summary.json")
        _verify_immutable(directory, manifest)
        if _sha(directory / "manifest.json") != manifest_hash or _sha(directory / "train.csv") != masking["masked_train_sha256"]:
            raise RuntimeError("Обучение изменило manifest или маскированный train")
        if summary.get("status") != "completed":
            raise RuntimeError("Фаза не завершена: " + str(summary.get("status")))
        if phase == "confirm" and (summary.get("score") is not None or not summary.get("label_blinded")):
            raise RuntimeError("Dummy score confirm не должен попадать в результаты")
        result = {"directory": str(directory), "phase_directory": str(phase_directory), "phase": phase,
                  "summary_path": str(phase_directory / "summary.json"), "prediction_path": str(phase_directory / "predictions.npz"),
                  "split_path": str(phase_directory / "split.npz"), "result_path": str(phase_directory / "result.json"),
                  "resource_audit_path": str(phase_directory / "runtime_isolation_attestation.json"),
                  "summary": summary, "metrics": summary.get("metrics", {}), "resource_audit": attestation,
                  "host_elapsed_seconds": time.monotonic() - started}
        _write_json(phase_directory / "host_result.json", result)
        return result
    except Exception as error:
        _write_json(phase_directory / "host_failure.json", {"status": "failed", "error_type": type(error).__name__,
                    "error": str(error)[-1800:], "phase": phase, "elapsed_seconds": time.monotonic() - started})
        raise
    finally:
        if created:
            try:
                subprocess.run(["docker", "rm", "--force", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False)
            except (OSError, subprocess.TimeoutExpired):
                # Основная ошибка остаётся в host_failure; имя контейнера есть в audit.
                pass
