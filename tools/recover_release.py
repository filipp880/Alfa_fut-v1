"""Восстановить отчёт хоста из успешно завершённого Docker, не повторяя fit."""
from pathlib import Path
import argparse
from datetime import datetime
import hashlib
import json
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    args = parser.parse_args()
    case = Path(args.case).resolve()
    destination = case / "release_result.json"
    if destination.exists():
        raise ValueError("Существующий результат не перезаписывается")
    manifest = json.loads((case / "release_manifest.json").read_text(encoding="utf-8"))
    info = json.loads(subprocess.check_output(
        ["docker", "inspect", manifest["container"]], text=True, encoding="utf-8"))[0]
    state = info["State"]
    if state["Running"] or state["ExitCode"] != 0 or state["OOMKilled"]:
        raise ValueError("Контейнер не завершился успешно")
    archive_hash = hashlib.sha256((case / "source.zip").read_bytes()).hexdigest()
    if archive_hash != manifest["archive_sha256"]:
        raise ValueError("Изменён проверенный ZIP")
    summary_path = sorted(case.glob(".alfa_runs/*/summary.json"))[-1]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] != "completed" or summary.get("fallback_only") or not summary.get("refit_completed"):
        raise ValueError("Не завершён полный агент/refit")
    elapsed = (datetime.fromisoformat(state["FinishedAt"].replace("Z", "+00:00")) -
               datetime.fromisoformat(state["StartedAt"].replace("Z", "+00:00"))).total_seconds()
    logs = subprocess.check_output(["docker", "logs", manifest["container"]], stderr=subprocess.STDOUT)
    (case / "console_recovered.log").write_bytes(logs)
    recovery = {"reason": "Хостовый attach прерван; контейнер самостоятельно завершился с exit 0",
                "container": manifest["container"], "state": state,
                "summary": str(summary_path.relative_to(case)),
                "elapsed_definition": "Docker StartedAt → FinishedAt; не хостовый monotonic",
                "no_training_repeated": True}
    (case / "host_recovery.json").write_text(json.dumps(recovery, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    record = {"directory": str(case), "archive_sha256": archive_hash, "host_seconds": elapsed,
              "elapsed_source": "docker_state_recovered", "status": summary["status"],
              "fallback_only": summary["fallback_only"], "refit_completed": summary["refit_completed"],
              "selected": summary["selected"], "weights": summary["weights"],
              "score": summary.get("confirmation_score")}
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
