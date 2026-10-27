"""Полный запуск заданного выпускного ZIP в отдельном офлайн-контейнере."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from isolated_trials import _audit_container
from run_kaggle import audit_dependencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--zip", required=True)
    parser.add_argument("--time-limit", type=int, default=10800)
    parser.add_argument("--max-experiments", type=int, default=18)
    parser.add_argument("--no-oof", action="store_true")
    parser.add_argument("--cpus", type=int, default=8, choices=range(1,9))
    parser.add_argument("--memory-gb", type=int, default=29, choices=range(1,30))
    args = parser.parse_args()
    if not 60 <= args.time_limit <= 10800: raise ValueError("Лимит ТЗ 60..10800с")
    archive = Path(args.zip).resolve(); source = Path(args.task).resolve()
    tag = uuid.uuid4().hex[:8]
    case = ROOT / "runs" / "release" / (source.parent.name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + tag)
    case.mkdir(parents=True)
    for name in ("task.md", "train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv"):
        shutil.copyfile(source.parent / name, case / name)
    shutil.copyfile(archive, case / "source.zip")
    with zipfile.ZipFile(archive) as bundle:
        oof_available = "oof_ensemble.py" in bundle.namelist()
        if bundle.testzip() or len(bundle.namelist()) != len(set(bundle.namelist())):
            raise ValueError("Повреждённый ZIP")
        for name in bundle.namelist():
            if Path(name).name != name or "/" in name or "\\" in name or not (name.endswith(".py") or name == "requirements.txt"):
                raise ValueError("Неплоский выпускной ZIP")
        bundle.extractall(case)
    code_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    container = "alfa-release-" + tag
    print("Полный выпускной прогон: " + str(case), flush=True)
    manifest = {"archive": archive.name, "archive_sha256": code_sha, "image": "alfa-research:local",
        "task_source":str(source.parent.relative_to(ROOT)) if source.parent.is_relative_to(ROOT) else str(source.parent),
        "input_sha256":{name:hashlib.sha256((source.parent/name).read_bytes()).hexdigest()
            for name in ("task.md","train.csv","test.csv","data_dictionary.csv","sample_submission.csv")},
        "time_limit_seconds": args.time_limit, "max_experiments": args.max_experiments,
        "max_iterations_cap": 0, "llm_calls": 0, "container": container,
        "cpu_total":args.cpus,"memory_bytes":args.memory_gb*2**30,
        "oof_available":oof_available,"oof_enabled":oof_available and not args.no_oof}
    (case / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    image = subprocess.check_output(["docker", "image", "inspect", "alfa-research:local", "--format", "{{.Id}}"], text=True).strip()
    command = ["docker", "create", "--name", container, "--pull=never", "--network=none", "--cpus="+str(args.cpus),
        "--memory="+str(args.memory_gb)+"g", "--memory-swap="+str(args.memory_gb)+"g",
        "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--tmpfs", "/tmp:rw,size=2g",
        "--tmpfs", "/agent:rw,noexec,size=1m", "--mount", f"type=bind,source={case},target=/task", "--workdir", "/task",
        "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "HOME=/tmp/home", "--env", "XDG_CACHE_HOME=/tmp/cache",
        "--env", "HTTP_PROXY=", "--env", "HTTPS_PROXY=", "--env", "ALL_PROXY=", "--env", "NO_PROXY=",
        "--entrypoint", "python3", image, "/task/solution.py", "/task/task.md", "--time-limit", str(args.time_limit),
        "--max-experiments", str(args.max_experiments), "--max-iterations", "0", "--threads",str(args.cpus), "--no-llm"]
    if args.no_oof and oof_available:
        command.append("--no-oof")
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    info = json.loads(subprocess.check_output(["docker", "inspect", container], text=True))[0]
    audit = _audit_container(info, case, container,expected_cpus=args.cpus,expected_memory_bytes=args.memory_gb*2**30)
    (case / "resource_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    timer = threading.Timer(args.time_limit + 15, lambda: subprocess.run(["docker", "stop", "-t", "2", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    timer.daemon = True; started = time.monotonic(); timer.start()
    try:
        with (case / "console.log").open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(["docker", "start", "--attach", container], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace")
            checked = False
            for line in process.stdout:
                print(line, end="", flush=True); stream.write(line); stream.flush()
                if not checked:
                    audit_dependencies(container); checked = True
            code = process.wait(timeout=30)
        if code: raise RuntimeError("Выпускной процесс завершился с кодом " + str(code))
    finally:
        timer.cancel()
        subprocess.run(["docker", "rm", "--force", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    paths = sorted(case.glob(".alfa_runs/*/summary.json"))
    summary = json.loads(paths[-1].read_text(encoding="utf-8"))
    record = {"directory": str(case), "archive_sha256": code_sha, "host_seconds": time.monotonic()-started,
        "status": summary["status"], "fallback_only": summary.get("fallback_only"), "refit_completed": summary.get("refit_completed"),
        "selected": summary.get("selected"), "weights": summary.get("weights"), "score": summary.get("confirmation_score")}
    (case / "release_result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if summary["status"] != "completed" or summary.get("fallback_only") or not summary.get("refit_completed"):
        raise RuntimeError("Полный выпускной прогон не завершил модель/refit")


if __name__ == "__main__": main()
