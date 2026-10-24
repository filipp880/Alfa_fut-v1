"""Изолированный стенд CPython 3.12 без изменения системного Python.

Запуск: python tools/bootstrap_windows.py. Только инструмент разработки.
"""
from pathlib import Path
import hashlib
import io
import json
import subprocess
import sys
import urllib.request
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / ".tools" / "python312"
    target.mkdir(parents=True, exist_ok=True)
    executable = target / "python.exe"
    if not executable.exists():
        url = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip"
        print("Downloading official CPython 3.12.10", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(target)
        (target / "python312._pth").write_text("python312.zip\n.\nLib/site-packages\nimport site\n")
    packages = target / "Lib" / "site-packages"
    if not (packages / "pip").exists():
        with urllib.request.urlopen("https://pypi.org/pypi/pip/json", timeout=30) as response:
            metadata = json.load(response)
        wheel = next(x for x in metadata["urls"] if x["filename"].endswith("py3-none-any.whl"))
        with urllib.request.urlopen(wheel["url"], timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != wheel["digests"]["sha256"]:
            raise ValueError("Wheel checksum mismatch")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(packages)
    return subprocess.call([str(executable), "-m", "pip", "--isolated", "install",
                            "--index-url", "https://pypi.org/simple", "--disable-pip-version-check",
                            "--no-warn-script-location", "-r", str(root / "requirements-dev.txt")])


if __name__ == "__main__":
    raise SystemExit(main())
