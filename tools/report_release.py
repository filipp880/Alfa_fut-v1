"""Ручной пересчёт полных выпускных ZIP и проверка фактического refit/CSV."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="1"
import numpy as np
import pandas as pd
from run_sgr_study import score,target
from task_reader import load_task
from submission import make_submission


def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):
    if b is None or not np.isclose(a,b,atol=1e-10,rtol=1e-10):raise ValueError(f"Метрика не совпала {a} != {b}")


def analyze(case):
    case=Path(case).resolve();release=read(case/"release_result.json");manifest=read(case/"release_manifest.json")
    work=sorted(case.glob(".alfa_runs/*/summary.json"))[-1].parent;s=read(work/"summary.json")
    if s["status"]!="completed" or s.get("fallback_only") or not s.get("refit_completed"):raise ValueError("Неполный модельный прогон")
    if sha(case/"source.zip")!=release["archive_sha256"] or manifest["archive_sha256"]!=release["archive_sha256"]:raise ValueError("ZIP изменён")
    with zipfile.ZipFile(case/"source.zip") as archive:
        if archive.testzip():raise ValueError("Повреждён ZIP")
        for name in archive.namelist():
            if "/" in name or "\\" in name or archive.read(name)!=(case/name).read_bytes():raise ValueError("Исходники отличаются")
    if sha(case/"requirements.txt")!="9173b0388532fbd2a709e4f4b3029e3b9ccab685fce43a85261ffa1ba95efbe6":raise ValueError("Изменены зависимости")
    dependency=read(case/"dependency_audit.json")
    if dependency["mismatches"]:raise ValueError("Установлены неверные версии")
    task,train,test,_,_=load_task(case/"task.md");y=target(task,train)
    original=ROOT/manifest["task_source"] if "task_source" in manifest else ROOT/"runs"/"public"/task.problem_type
    for name in ("task.md","train.csv","test.csv","data_dictionary.csv","sample_submission.csv"):
        if sha(original/name)!=sha(case/name):raise ValueError("Изменены исходные данные "+name)
        if manifest.get("input_sha256",{}).get(name,sha(case/name))!=sha(case/name):raise ValueError("Изменён снимок входных данных")
    arrays={};records={};count=0;failed=[];jobs=[]
    for path in sorted(work.glob("*/controller_result.json")):
        r=read(path);req=read(path.parent/"request.json");phase=r["phase"];key=r["config_key"]
        if r["status"]!="ok":failed.append({"phase":phase,"status":r["status"]});continue
        with np.load(work/Path(req["split_path"]).name,allow_pickle=False) as split:tr,va=split["train"],split["valid"]
        if np.intersect1d(tr,va).size:raise ValueError("Пересечение split")
        if task.classification and len(va):
            g=pd.util.hash_pandas_object(train[task.feature_columns],index=False).to_numpy()
            if np.intersect1d(g[tr],g[va]).size:raise ValueError("Пересечение одинаковых X")
        if len(va) and s["profile"].get("time_column"):
            timecol=s["profile"]["time_column"]
            if train.iloc[tr][timecol].max()>=train.iloc[va][timecol].min():raise ValueError("Нарушена хронология")
        if phase in {"confirm","refit"}:
            if not req["refit"] or req["fixed_iterations"]!=r["best_iterations"] or r.get("evaluation_used_for_fit",False):
                raise ValueError("Confirm/refit не зафиксирован")
            source=records.get(("search",key))
            if source is None or source["best_iterations"]!=req["fixed_iterations"]:raise ValueError("Количество итераций не из search")
        with np.load(path.parent/"predictions.npz",allow_pickle=False) as p:
            np.testing.assert_array_equal(p["valid_indices"],va);valid,pred=p["valid"].copy(),p["test"].copy()
        if len(valid)!=len(va) or len(pred)!=len(test) or not np.isfinite(valid).all() or not np.isfinite(pred).all():raise ValueError("Неверные predictions")
        if len(va):close(score(task,y[va],valid),r["score"])
        arrays[(phase,key)]=(valid,pred,va);records[(phase,key)]=r
        fits=r.get("component_fits",1);count+=fits
        jobs.append({"phase":phase,"model":r["config"]["model"],"score":r.get("score"),"fits":fits,"seconds":r["wall_seconds"]})
    selected_keys=[]
    for component in s["selected"]:
        match=next((k for (p,k),r in records.items() if p==component["phase"] and r["config"]==component["config"]),None)
        if match is None:raise ValueError("Не найдена выбранная модель")
        selected_keys.append(match)
    weights=s["weights"];confirm=sum(w*arrays[("confirm",k)][0] for w,k in zip(weights,selected_keys))
    indices=arrays[("confirm",selected_keys[0])][2];confirm_score=score(task,y[indices],confirm)
    if s.get("confirmation_score") is not None:close(confirm_score,s["confirmation_score"])
    final=sum(w*arrays[("refit",k)][1] for w,k in zip(weights,selected_keys))
    expected=make_submission(task,test,final);actual=pd.read_csv(case/"submission.csv",dtype={c:str for c in task.id_columns},keep_default_na=False)
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,rtol=1e-12,atol=1e-12)
    f=read(work/"finalists.json");whole_frozen="selected_plan" in f
    if whole_frozen:
        plan=s["selected_plan"]
        allowed=[f["selected_plan"],f["baseline_plan"]]
        if f.get("oof_selected") is True:
            allowed.append(f["legacy_selected_plan"])
        if plan not in allowed:raise ValueError("Новый план выбран по confirm")
        if plan["members"]!=selected_keys or plan["weights"]!=weights:raise ValueError("Финальный состав изменён")
        if "legacy_selected_plan" in f:
            def plan_score(p):
                if any(("confirm",k) not in arrays for k in p["members"]):return None
                values=sum(w*arrays[("confirm",k)][0] for w,k in zip(p["weights"],p["members"]))
                return score(task,y[indices],values)
            def gain(a,b):return (a-b if task.maximize else b-a)>.001*max(1,abs(b))
            guard=f["legacy_selected_plan"];control=f["legacy_baseline_plan"]
            gs,cs=plan_score(guard),plan_score(control)
            if cs is not None and (gs is None or (guard!=control and not gain(gs,cs))):guard=control
            proposal=f["selected_plan"];gs,ps=plan_score(guard),plan_score(proposal)
            expected=proposal if f["oof_selected"] and gs is not None and ps is not None and gain(ps,gs) else guard
            if plan!=expected:raise ValueError("Нарушено правило независимого подтверждения OOF/контроля")
        for (phase,key),r in records.items():
            if phase=="confirm" and (work/"finalists.json").stat().st_mtime_ns>Path(r["directory"].replace("/task",str(case),1)).joinpath("request.json").stat().st_mtime_ns:
                raise ValueError("План зафиксирован после confirm")
        manifest_csv=read(case/"submission.manifest.json")
        if manifest_csv["status"]!="ok" or manifest_csv["submission"]["sha256"]!=sha(case/"submission.csv"):raise ValueError("CSV-манифест неверен")
        search_train=np.load(work/"split_search.npz")["train"]
        profile=read(work/"profile.json")
        if profile.get("profile_scope")!="search_train_only" or profile["train_rows"]!=len(search_train):raise ValueError("Профиль не из search-train")
        if task.classification:
            counts=train.iloc[search_train][task.target].value_counts().to_dict()
            if profile["class_counts"]!=counts:raise ValueError("Чужие class-counts в профиле")
        else:
            close(profile["target_summary"]["mean"],float(y[search_train].mean()))
    audit=read(case/"resource_audit.json")
    if audit["cpu_total"]>8 or audit["memory_bytes_total"]>32000000000 or audit["network"]!="none":raise ValueError("Нарушены ресурсы")
    events=[json.loads(line) for line in (work/"experiments.jsonl").read_text(encoding="utf-8").splitlines()]
    comparison=next((e for e in reversed(events) if e["event"]=="frozen_plan_confirmed"),{})
    comparison={k:comparison[k] for k in ("planned_score","legacy_score","baseline_score","guard_score") if k in comparison}
    from audit_oof_release import analyze as audit_oof
    oof=audit_oof(case)
    oracle_score=None
    if (original/"oracle_test.npz").exists():
        if (case/"oracle_test.npz").exists():raise ValueError("Тестовый oracle попал к агенту")
        oracle=np.load(original/"oracle_test.npz",allow_pickle=False)["target"]
        if task.classification:
            axis={v:i for i,v in enumerate(task.classes)}
            oracle=np.array([axis[v] for v in oracle],dtype=int)
        oracle_score=score(task,oracle,final)
    return {"directory":str(case.relative_to(ROOT)),"task":task.problem_type,"metric":task.metric,
        "archive_sha256":release["archive_sha256"],"confirm_score":confirm_score,"weights":weights,
        "selected_models":[c["config"]["model"] for c in s["selected"]],"worker_jobs":len(jobs),
        "base_ml_fits":count,"ml_fits":count+(oof.get("completed_fits") or 0),"oof":oof,
        "oracle_test_score":oracle_score,"failures":failed,
        "confirmation_comparison":comparison,
        "host_seconds":release["host_seconds"],"elapsed_source":release.get("elapsed_source","host_monotonic"),
        "agent_seconds":s["elapsed_seconds"],"cpu":audit["cpu_total"],"memory_limit_bytes":audit["memory_bytes_total"],
        "csv_rows":len(actual),"csv_sha256":sha(case/"submission.csv"),"whole_plan_frozen_before_confirm":whole_frozen,
        "profile_search_train_only":whole_frozen,"refit_completed":True,"no_iteration_cap":manifest["max_iterations_cap"]==0,
        "jobs":jobs,"checks":{"manual_metrics":True,"source_zip_intact":True,"original_data_intact":True,"exact_indices":True,
            "fixed_confirm_and_refit":True,"csv_matches_full_refit":True,"approved_dependencies":True}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("cases",nargs="+");parser.add_argument("--output",default="docs/RELEASE_RESULTS.json")
    args=parser.parse_args();results=[analyze(p) for p in args.cases]
    Path(args.output).write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for r in results:print(r["task"],r["selected_models"],r["confirm_score"],r["host_seconds"])


if __name__=="__main__":main()
