"""Создать рабочие копии материалов, не изменяя исходную папку."""
from pathlib import Path
import argparse
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Каталог data_for_participants")
    parser.add_argument("--destination", default="runs/public")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    destination = Path(args.destination).resolve()
    for task in source.rglob("task.md"):
        text = task.read_text(encoding="utf-8-sig").lower()
        name = "classification" if "macro f1" in text else "regression" if "rmse" in text else task.parent.name
        target = destination/name
        target.mkdir(parents=True, exist_ok=True)
        for name in ("task.md", "train.csv", "test.csv", "sample_submission.csv", "data_dictionary.csv"):
            original = task.parent/name
            if original.exists():
                shutil.copyfile(original, target/name)
        print(target)


if __name__ == "__main__":
    main()
