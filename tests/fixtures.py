from pathlib import Path
import numpy as np
import pandas as pd


def make_task(root: Path, classification=False, probabilities=False, temporal=False, n=240):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    x = rng.normal(size=n+50)
    frame = pd.DataFrame({"id": [f"{i:06d}" for i in range(n+50)], "x": x,
                          "category": np.where(x > 0, "positive", "negative")})
    frame.loc[::19, "category"] = np.nan
    if temporal:
        frame["period_year"] = np.concatenate([np.repeat(np.arange(1, 6), n//5), np.full(50, 6)])
    target = "category_to_predict" if classification else "response"
    y = np.where(x > .5, "high", np.where(x < -.5, "low", "middle")) if classification else 4*x+20+rng.normal(0, .08, n+50)
    train, test = frame.iloc[:n].copy(), frame.iloc[n:].copy()
    train[target] = y[:n]
    test.loc[test.index[:2], "category"] = "new_category"
    train.to_csv(root/"train.csv", index=False)
    test.to_csv(root/"test.csv", index=False)
    if probabilities:
        sample = pd.DataFrame({"id": test.id, "middle": .3, "high": .3, "low": .4})
    else:
        sample = pd.DataFrame({"id": test.id, target: "middle" if classification else 0})
    sample.to_csv(root/"sample_submission.csv", index=False)
    rows = [{"feature": c, "role": "id" if c == "id" else "feature",
             "dtype": "category" if c == "category" else "numeric", "description_ru": c} for c in frame]
    rows.append({"feature": target, "role": "target", "dtype": "category" if classification else "numeric"})
    pd.DataFrame(rows).to_csv(root/"data_dictionary.csv", index=False)
    metric = "Log Loss" if probabilities else "Macro F1" if classification else "RMSE"
    (root/"task.md").write_text(f"# Задача\nОсновная метрика — **{metric}**.\n"
        f"Таргет `{target}`. `missing/path/train.csv`, `missing/path/test.csv`.\n"
        "Файл submission.csv. Порядок строк совпадает с test.csv.\n", encoding="utf-8")
    return root/"task.md"
