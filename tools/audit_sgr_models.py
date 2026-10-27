"""Проверка реальных сохранённых CatBoost-моделей завершённого SGR-теста."""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()
    run = Path(args.run).resolve()
    if not run.is_relative_to(ROOT / "runs" / "sgr"):
        raise ValueError("Ожидается локальный runs/sgr/<id>")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed" or len(summary["arms"]) != 8:
        raise ValueError("Исследование ещё не завершено")
    records = []
    for arm in summary["arms"]:
        if arm["status"] != "completed" or arm["executed_config"]["model"] != "catboost":
            raise ValueError("Ожидаются восемь успешно выполненных CatBoost-предложений этого теста")
        for job in arm["jobs"]:
            directory = (run / job["directory"]).resolve()
            if not directory.is_relative_to(run):
                raise ValueError("Модель вне текущего исследования")
            paths = list(directory.rglob("model.pkl"))
            if len(paths) != 1:
                raise ValueError("Нет единственной сохранённой модели фазы")
            path = paths[0]
            # Файлы произведены доверенным локальным worker; внешние pickle не принимаются.
            with path.open("rb") as stream:
                bundle = pickle.load(stream)
            model = bundle["model"]
            count = model.tree_count_
            seed = model.get_all_params()["random_seed"]
            if count != arm["frozen"]["best_iterations"] or seed != arm["seed"]:
                raise ValueError("Фактические деревья/seed отличаются от frozen")
            if bundle["config"] != arm["executed_config"]:
                raise ValueError("Сохранённая конфигурация отличается от выполненной")
            records.append({"label": arm["label"], "phase": job["phase"],
                            "actual_tree_count": count, "actual_seed": seed,
                            "model_path": path.relative_to(run).as_posix(),
                            "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if len(records) != 24:
        raise ValueError("Ожидаются все 24 сохранённые модели")
    result = {"status": "verified", "models": records, "new_fits": 0, "new_api_calls": 0}
    (run / "saved_model_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Проверены 24 фактически сохранённые модели: counts, seed, config и SHA-256")


if __name__ == "__main__":
    main()
