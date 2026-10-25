"""Изолированный офлайн-прогон сравнительного исследования в пределах ТЗ."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from package_solution import MODULES


def package(path):
    members = [name for name in MODULES if name != "solution.py"]+[
        "novel_models.py", "novel_worker.py", "unconventional.py"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        content = {name: (ROOT/name).read_bytes() for name in members}
        content["solution.py"] = b"from unconventional import main\nraise SystemExit(main())\n"
        for name, data in content.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    if path.stat().st_size > 100*1024**2:
        raise ValueError("Архив превышает 100 МБ")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--image", default="alfa-research:local")
    parser.add_argument("--time-limit", type=int, default=10800)
    parser.add_argument("--trial-seconds", type=int, default=360)
    parser.add_argument("--control-trials", type=int, default=8)
    args = parser.parse_args()
    if not 30 <= args.time_limit <= 10800:
        raise ValueError("Лимит целого запуска превышает ТЗ")
    source = Path(args.task).resolve()
    tag = uuid.uuid4().hex[:10]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    destination = ROOT/"runs"/"unconventional"/f"{source.parent.name}_{stamp}_{tag}"
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("task.md", "train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv"):
        shutil.copyfile(source.parent/name, destination/name)
    archive = package(ROOT/"dist"/"alfa_unconventional.zip")
    with zipfile.ZipFile(archive) as z:
        if any("/" in name or "\\" in name for name in z.namelist()):
            raise ValueError("Вложенные файлы в ZIP")
        z.extractall(destination)
    container = "alfa-novel-"+tag
    manifest = {"container": container, "task_directory": str(destination),
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "network": "none", "cpu_limit": 8, "memory_limit_bytes": 29*1024**3,
                "time_limit_seconds": args.time_limit, "image": args.image}
    (destination/"docker_objects.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    command = ["docker", "run", "--name", container, "--network=none", "--cpus=8", "--memory=29g",
               "--memory-swap=29g", "--read-only", "--tmpfs", "/tmp:rw,size=2g", "--mount",
               f"type=bind,source={destination},target=/task", "--env", "PYTHONDONTWRITEBYTECODE=1",
               "--entrypoint", "python3", args.image, "/task/solution.py", "task.md",
               "--time-limit", str(args.time_limit), "--trial-seconds", str(args.trial_seconds),
               "--control-trials", str(args.control_trials)]
    print("Каталог исследования: "+str(destination), flush=True)
    try:
        with (destination/"console.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            audited = False
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line); log.flush()
                if not audited:
                    host = json.loads(subprocess.check_output(
                        ["docker", "inspect", "--format", "{{json .HostConfig}}", container]))
                    audit = {"cpu_limit": host["NanoCpus"]/1e9, "memory_bytes": host["Memory"],
                             "memory_swap_bytes": host["MemorySwap"], "network": host["NetworkMode"],
                             "readonly_root": host["ReadonlyRootfs"]}
                    if not (audit["cpu_limit"] <= 8 and audit["memory_bytes"] <= 32000000000
                            and audit["network"] == "none"):
                        raise RuntimeError("Ресурсные ограничения нарушены")
                    (destination/"resource_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
                    audited = True
            code = process.wait(timeout=args.time_limit+60)
        if code:
            raise RuntimeError(f"Исследование завершилось с кодом {code}")
    finally:
        subprocess.run(["docker", "rm", "--force", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    summaries = sorted(destination.glob(".unconventional/*/summary.json"))
    result = json.loads(summaries[-1].read_text(encoding="utf-8")) if summaries else {}
    if result.get("status") != "completed" or any(
            result.get("methods", {}).get(name, {}).get("status") != "ok"
            for name in ("control", "extra_trees", "kernel", "retrieval")):
        raise RuntimeError("Не все гипотезы прошли подтверждающую проверку")
    print(json.dumps({"winner": result["winner"], "winner_score": result["winner_score"],
                      "directory": str(destination)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
