"""Финальный аудит артефактов исследования без новых обучений и API."""
from pathlib import Path
import ast
from datetime import datetime, timezone
import hashlib
import json
import re
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
from package_solution import MODULES


def main():
    import numpy as np
    import pandas as pd
    from task_reader import load_task
    archive=ROOT/"dist"/"alfa_solution.zip"
    requirements_sha="9173b0388532fbd2a709e4f4b3029e3b9ccab685fce43a85261ffa1ba95efbe6"
    assert hashlib.sha256((ROOT/"requirements.txt").read_bytes()).hexdigest()==requirements_sha
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist()==list(MODULES) and bundle.testzip() is None
        assert all("/" not in n and "\\" not in n for n in bundle.namelist())
        for name in MODULES:
            assert bundle.read(name)==(ROOT/name).read_bytes(),name
            if name.endswith(".py"):
                ast.parse(bundle.read(name).decode("utf-8"),filename=name)
    assert archive.stat().st_size<100*1024**2
    backup=ROOT/"dist"/"alfa_solution_faedd2a.zip"
    assert hashlib.sha256(backup.read_bytes()).hexdigest()=="079a5f610baa616b10c0058e4c4e4860705f372f7b75f0b325d9a74c9b795a79"
    tests={}
    for platform in ("windows","linux"):
        path=ROOT/"runs"/"tests"/f"kaggle_final_{platform}.log"
        body=path.read_text(encoding="utf-8-sig",errors="replace")
        # PowerShell 5 может писать перенаправленный журнал в UTF-16.
        if "\x00" in body:
            body=path.read_text(encoding="utf-16")
        match=re.search(r"Ran (\d+) tests in ([\d.]+)s",body)
        assert match and int(match[1])==69 and "\nOK" in body and "FAILED (" not in body,platform
        tests[platform]={"tests":int(match[1]),"seconds":float(match[2]),
                         "skipped":3 if "OK (skipped=3)" in body else 0,
                         "log":str(path.relative_to(ROOT))}
    cases=json.loads((ROOT/"docs"/"KAGGLE_RESULTS.json").read_text(encoding="utf-8"))
    checked=[]
    for record in cases:
        case=ROOT/record["directory"]
        task,train,test,sample,_=load_task(case/"task.md")
        output=pd.read_csv(case/"submission.csv",dtype={c:"string" for c in task.id_columns})
        assert list(output)==task.output_columns and len(output)==len(test)
        for column in task.id_columns:
            assert output[column].astype(str).tolist()==test[column].astype(str).tolist()
        predictions=[c for c in task.output_columns if c not in task.id_columns]
        if task.classification:
            assert set(output[predictions[0]].astype(str))<=set(map(str,task.classes))
        else:
            assert np.isfinite(output[predictions].to_numpy(dtype=float)).all()
        assert record["final_refit_completed"] and record["archive_audit"]["verified_archive"]
        assert record["dependency_audit"]["mismatches"]==[] and all(record["checks"].values())
        checked.append({"task":record["task"],"directory":record["directory"],"rows":len(output),
                        "score":record["selected"]["confirmation_score"],"final_refit_completed":True,
                        "experiment_count":record["experiment_count"],"elapsed_seconds":record["elapsed_seconds"]})
    for name in ("KAGGLE_REPEAT_RESULTS.json","KAGGLE_CLASSIFICATION_REPEAT_RESULTS.json"):
        record=json.loads((ROOT/"docs"/name).read_text(encoding="utf-8"))
        assert record["status"]=="completed" and all(record["checks"].values()),name
    syntax_files=list(ROOT.glob("*.py"))+list((ROOT/"tools").glob("*.py"))+list((ROOT/"tests").glob("*.py"))
    for path in syntax_files:
        ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
    audit={"status":"passed","checked_at_utc":datetime.now(timezone.utc).isoformat(),
           "requirements_sha256":requirements_sha,"archive":{"bytes":archive.stat().st_size,
           "sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),"python_modules":len(MODULES)-1,
           "flat":True,"matches_sources":True,"backup_preserved":True},"tests":tests,"cases":checked,
           "core_experiments":sum(c["experiment_count"] for c in checked),"additional_seed_fits":10,
           "syntax_files_checked":len(syntax_files),"production_policy_changed":False,
           "note":"Усреднённая история остаётся post-hoc исследовательским результатом; штатные модели не заменены."}
    path=ROOT/"docs"/"kaggle_release_validation_20261025.json"
    path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
