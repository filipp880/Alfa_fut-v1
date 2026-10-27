"""Внешний watchdog времени и памяти для каждого процесса обучения."""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import psutil

from contracts import ExperimentSpec, TaskSpec
from experiment_store import atomic_json


def cpu_limit() -> int:
    count = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        count = min(count, len(os.sched_getaffinity(0)))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, int(quota)//int(period)))
    except (OSError, ValueError):
        pass
    return max(1, min(8, count))


def terminate_tree(process: subprocess.Popen):
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.Error:
                pass
        if process.poll() is None:
            process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait(timeout=3)
    for child in children:
        try:
            if child.is_running():
                child.kill()
        except psutil.Error:
            pass


class Budget:
    def __init__(self, started: float, seconds: float, threads: int, memory_mb: int):
        self.started = started
        self.seconds = min(float(seconds), 10800)
        self.deadline = started + self.seconds
        self.finish_margin = min(30.0, max(2.0, self.seconds * .015))
        self.threads = max(1, min(threads, cpu_limit()))
        self.memory_mb = max(128, min(memory_mb, 26000))

    def remaining(self) -> float:
        return max(0.0, self.deadline-time.monotonic())

    def until(self, fraction: float) -> float:
        return max(0.0, self.started+self.seconds*fraction-time.monotonic())


class Runner:
    def __init__(self, directory: Path, task: TaskSpec, profile: dict, budget: Budget,
                 worker_module: str = "worker"):
        if worker_module not in {"worker", "strategy_worker", "novel_worker", "kaggle_worker"}:
            raise ValueError("Недопустимый исполнитель")
        self.directory, self.task, self.profile, self.budget = directory, task, profile, budget
        self.worker_module = worker_module
        self.code_root = Path(__file__).resolve().parent
        self.counter = 0

    def run(self, config: ExperimentSpec, split_path: Path, phase: str, timeout: float,
            refit: bool = False, fixed_iterations: int | None = None,
            job_options: dict | None = None) -> dict:
        config.validate(self.task)
        self.counter += 1
        work = self.directory / f"{self.counter:03d}_{phase}_{config.key}"
        work.mkdir(parents=True, exist_ok=False)
        request = {"task": self.task.to_dict(), "config": config.to_dict(), "profile": self.profile,
                   "split_path": str(split_path.resolve()), "threads": self.budget.threads,
                   "refit": refit, "fixed_iterations": fixed_iterations,
                   "job_options": job_options or {}}
        atomic_json(work / "request.json", request)
        # Генерация исполняемой программы из фиксированного шаблона и проверенного DSL.
        program = ("import sys\n" + f"sys.path.insert(0, {str(self.code_root)!r})\n"
                   + f"from {self.worker_module} import main\n"
                   + f"raise SystemExit(main({str(work / 'request.json')!r}))\n")
        (work / "experiment.py").write_text(program, encoding="utf-8")
        allowed = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "LD_LIBRARY_PATH",
                   "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH"}
        environment = {k: v for k, v in os.environ.items() if k.upper() in allowed}
        environment.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                           PYTHONHASHSEED=str(config.seed), OMP_NUM_THREADS=str(self.budget.threads),
                           MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                           VECLIB_MAXIMUM_THREADS="1",
                           MPLCONFIGDIR=str(self.directory / ".cache" / "matplotlib"))
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        maximum = min(timeout, self.budget.remaining()-self.budget.finish_margin)
        if maximum <= 0:
            return {"status": "skipped", "reason": "Нет времени", "directory": str(work)}
        request["fit_time_limit_seconds"] = max(.5, maximum-max(4, min(20, maximum*.15)))
        atomic_json(work / "request.json", request)
        total_memory = 32*2**30
        for location in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            try:
                limit = int(Path(location).read_text().strip())
                if limit > 0:
                    total_memory = min(total_memory, limit)
            except (OSError, ValueError):
                pass
        memory_limit = min(self.budget.memory_mb*2**20,
                           total_memory*.9-psutil.Process().memory_info().rss,
                           psutil.virtual_memory().available*.85)
        if memory_limit < 64*2**20:
            return {"status": "skipped", "reason": "Недостаточно свободной памяти", "directory": str(work)}
        started, peak, stopped = time.monotonic(), 0, None
        with (work / "stdout.log").open("wb") as stdout, (work / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen([sys.executable, "-B", "-u", str(work / "experiment.py")],
                                       cwd=Path(self.task.task_path).parent, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, **kwargs)
            try:
                watched = psutil.Process(process.pid)
                while process.poll() is None:
                    if time.monotonic()-started > maximum:
                        stopped = "timeout"
                        break
                    try:
                        resident = watched.memory_info().rss
                        for child in watched.children(recursive=True):
                            try:
                                resident += child.memory_info().rss
                            except psutil.Error:
                                pass
                        peak = max(peak, resident)
                        if resident > memory_limit:
                            stopped = "memory_limit"
                            break
                    except psutil.NoSuchProcess:
                        break
                    time.sleep(.2)
            finally:
                if stopped or process.poll() is None:
                    terminate_tree(process)
                process.wait()
        if stopped:
            result = {"status": stopped, "reason": stopped}
        elif (work / "result.json").exists():
            result = json.loads((work / "result.json").read_text(encoding="utf-8"))
            if process.returncode and result.get("status") == "ok":
                result = {"status": "failed", "reason": "Ненулевой код процесса"}
        else:
            result = {"status": "failed", "reason": f"Процесс завершился без результата: {process.returncode}"}
        result.update(directory=str(work), peak_memory_mb=round(peak/2**20, 2),
                      effective_memory_limit_mb=round(memory_limit/2**20, 2),
                      wall_seconds=time.monotonic()-started, phase=phase,
                      config=config.to_dict(), config_key=config.key)
        atomic_json(work / "controller_result.json", result)
        return result
