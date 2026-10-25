"""Детерминированная сборка и проверка плоского ZIP для платформы."""
from __future__ import annotations
import argparse
import ast
import hashlib
from pathlib import Path
import zipfile


MODULES = (
    "solution.py", "contracts.py", "task_reader.py", "data_io.py", "validation.py",
    "metrics.py", "submission.py", "preprocessing.py", "models.py", "worker.py",
    "runtime.py", "experiment_store.py", "research.py", "llm_client.py",
    "novel_models.py", "novel_worker.py", "requirements.txt",
)


def build(root: Path, destination: Path) -> Path:
    root, destination = root.resolve(), destination.resolve()
    for name in MODULES:
        source = root/name
        if not source.is_file():
            raise ValueError(f"Отсутствует обязательный файл: {name}")
        if name.endswith(".py"):
            ast.parse(source.read_text(encoding="utf-8"), filename=name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in MODULES:
            entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, (root/name).read_bytes())
    if temporary.stat().st_size > 100*1024*1024:
        temporary.unlink()
        raise ValueError("ZIP превышает 100 МБ")
    with zipfile.ZipFile(temporary) as archive:
        if archive.namelist() != list(MODULES) or archive.testzip() is not None:
            raise ValueError("Неверный состав или целостность ZIP")
        if any("/" in name or "\\" in name for name in archive.namelist()):
            raise ValueError("В ZIP есть вложенные файлы")
    temporary.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix+".sha256").write_text(digest+"  "+destination.name+"\n", encoding="ascii")
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist/alfa_solution.zip")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = build(root, root/args.output)
    print(f"Архив: {result}; размер: {result.stat().st_size} байт")


if __name__ == "__main__":
    main()
