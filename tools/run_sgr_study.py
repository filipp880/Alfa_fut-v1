"""Реальное парное SGR-сравнение: свежие контексты, одинаковый DSL и полные fit."""
from pathlib import Path
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
import numpy as np
from data_io import profile_data
from experiment_store import Journal,atomic_json
from isolated_trials import create_trial,run_phase
from llm_client import LLMPlanner
from sgr_planner import SGRPlanner
from runtime import Budget
from submission import write_submission
from submission_audit import audit_submission
from task_reader import load_task
from validation import make_splits

GATEWAY="https://api.alfahack-gateway26.ru/v1"
MODEL="qwen/qwen3.8-27b"


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target(task,frame):
    if not task.classification:return frame[task.target].to_numpy(dtype=float)
    mapping={value:i for i,value in enumerate(task.classes)}
    return np.array([mapping[value] for value in frame[task.target]],dtype=int)


def score(task,y,p):
    if not task.classification:return float(np.sqrt(np.mean(np.square(np.asarray(y)-np.asarray(p)))))
    classes=len(task.classes);p=np.asarray(p)
    if p.shape!=(len(y),classes) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1):
        raise ValueError("Неверная ось классов или вероятности")
    matrix=np.bincount(y*classes+p.argmax(1),minlength=classes**2).reshape(classes,classes)
    denominator=matrix.sum(0)+matrix.sum(1)
    return float(np.divide(2*np.diag(matrix),denominator,out=np.zeros(classes),where=denominator>0).mean())


def improves(task,value,previous):
    return (value-previous if task.maximize else previous-value)>.001*max(1,abs(previous))


def reference(case,run,label):
    case=Path(case).resolve();summary_path=sorted(case.glob(".alfa_runs/*/summary.json"))[-1]
    summary=read(summary_path)
    if summary["status"]!="completed" or summary.get("fallback_only") or not summary.get("refit_completed"):
        raise ValueError("Основная версия ещё не прошла полный refit")
    plan=summary["selected_plan"];work=summary_path.parent
    task,train,test,_,dictionary=load_task(case/"task.md")
    y=target(task,train);outputs={};split_indices={}
    jobs={}
    for path in work.glob("*/controller_result.json"):
        r=read(path)
        if r["status"]=="ok":jobs[(r["phase"],r["config_key"])]=path.parent
    for phase in ("search","confirm","refit"):
        pieces=[]
        for key in plan["members"]:
            with np.load(jobs[(phase,key)]/"predictions.npz",allow_pickle=False) as data:
                indices=data["valid_indices"].copy()
                pieces.append((data["valid"].copy(),data["test"].copy()))
            if phase in split_indices:np.testing.assert_array_equal(indices,split_indices[phase])
            else:split_indices[phase]=indices
        outputs[phase]=tuple(sum(w*p[i] for w,p in zip(plan["weights"],pieces)) for i in (0,1))
    release=read(case/"release_result.json")
    if sha(case/"source.zip") != release["archive_sha256"]:
        raise ValueError("Проверенный выпускной ZIP изменён после основного прогона")
    profile,groups,times=profile_data(task,train,test,dictionary)
    standard=make_splits(task,y,groups,times)
    for name,split in zip(("search","confirm"),standard[:2]):np.testing.assert_array_equal(split.valid,split_indices[name])
    profile,_,_=profile_data(task,train.iloc[standard[0].train],test,dictionary)
    profile["time_column"]=read(work/"profile.json")["time_column"]
    source=run/"sources"/label;source.mkdir(parents=True)
    with zipfile.ZipFile(case/"source.zip") as archive:
        for name in archive.namelist():
            if Path(name).name!=name or not(name.endswith(".py") or name=="requirements.txt"):raise ValueError("Неверный исходный архив")
        archive.extractall(source)
        for name in archive.namelist():
            if archive.read(name)!=(case/name).read_bytes():
                raise ValueError("Распакованные исходники не совпадают с проверенным ZIP")
    for name in ("contracts.py","llm_client.py","sgr_planner.py"):
        if sha(source/name)!=sha(ROOT/name):raise ValueError("Планировщик изменён после проверки основного ZIP")
    wrapper_hash=sha(ROOT/"isolated_trial.py")
    shutil.copyfile(ROOT/"isolated_trial.py",source/"isolated_trial.py")
    metadata={"release_case":str(case.relative_to(ROOT)),"archive_sha256":release["archive_sha256"],"plan":plan,
        "train_sha256":sha(task.train_path),"test_sha256":sha(task.test_path),
        "search_score":score(task,y[split_indices["search"]],outputs["search"][0]),
        "confirm_score":score(task,y[split_indices["confirm"]],outputs["confirm"][0])}
    metadata["instrumentation_sha256"]={"isolated_trial.py":wrapper_hash}
    atomic_json(run/("reference_"+label+".json"),metadata)
    np.savez_compressed(run/("reference_"+label+".npz"),search=outputs["search"][0],confirm=outputs["confirm"][0],
        full_test=outputs["refit"][1],search_indices=split_indices["search"],confirm_indices=split_indices["confirm"])
    return {"task":task,"train":train,"test":test,"y":y,"profile":profile,"splits":standard,
        "outputs":outputs,"metadata":metadata,"sources":[source/n for n in sorted(p.name for p in source.iterdir())]}


class Recording:
    def request(self,system,payload,*,response_schema=None):
        if payload.get("history")!=[]:raise ValueError("В ограниченном парном тесте история обязана быть пустой")
        atomic_json(self.prompt_path,{"system":system,"payload":payload,"response_schema":response_schema,
            "other_policy_context":False,"previous_reports_in_context":False,
            "request_settings":{"response_format":None if response_schema is None else {"type":"json_schema","json_schema":{
                "name":"ml_research_step","strict":True,"schema":response_schema}},
                "temperature":.15,"max_tokens":1000,"max_retries":0,"enable_thinking":False,"model":MODEL,"gateway":GATEWAY}})
        return super().request(system,payload,response_schema=response_schema)


class Normal(Recording,LLMPlanner):pass
class Structured(Recording,SGRPlanner):pass


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--reference-regression",required=True)
    parser.add_argument("--reference-classification",required=True);parser.add_argument("--key-file",required=True)
    parser.add_argument("--token-limit",type=int,default=40000);args=parser.parse_args()
    if not 1000<=args.token_limit<=40000:raise ValueError("Токенный предел 1000..40000")
    key=Path(args.key_file).read_text(encoding="utf-8-sig").strip()
    if not key:raise ValueError("Пустой ключ")
    run=ROOT/"runs"/"sgr"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S");run.mkdir(parents=True)
    print("Реальный SGR-тест: "+str(run),flush=True)
    refs={"regression":reference(args.reference_regression,run,"regression"),
          "classification":reference(args.reference_classification,run,"classification")}
    if refs["regression"]["metadata"]["archive_sha256"]!=refs["classification"]["metadata"]["archive_sha256"]:
        raise ValueError("Нельзя сравнивать разные основные версии")
    summary={"status":"running","gateway":GATEWAY,"model":MODEL,"max_calls":8,"completion_limit":1000,
        "token_limit":args.token_limit,"calls":0,"tokens":0,"arms":[],"protocol_sha256":sha(ROOT/"docs/SGR_PROTOCOL.md"),
        "reference_archive_sha256":refs["regression"]["metadata"]["archive_sha256"]}
    def save():atomic_json(run/"summary.json",summary)
    save()
    order=[]
    for task_name in ("regression","classification"):
        for seed in (42,43):
            policies=("normal","sgr") if (task_name=="regression")== (seed==42) else ("sgr","normal")
            order.extend((task_name,seed,policy) for policy in policies)
    for task_name,seed,policy in order:
        ref=refs[task_name];task=ref["task"];started=time.monotonic()
        label=f"{task_name}_{policy}_{seed}";directory=run/"prompts"/label;directory.mkdir(parents=True)
        remaining_tokens=args.token_limit-summary["tokens"]
        if remaining_tokens<1000:summary["budget_exhausted"]=True;break
        journal=Journal(directory);budget=Budget(started,10800,8,24000)
        saved={name:os.environ.get(name) for name in ("OPENAI_API_KEY","OPENAI_BASE_URL","OPENAI_MODEL","ALFA_LLM_TOKEN_LIMIT")}
        try:
            os.environ.update(OPENAI_API_KEY=key,OPENAI_BASE_URL=GATEWAY,OPENAI_MODEL=MODEL,ALFA_LLM_TOKEN_LIMIT=str(remaining_tokens))
            planner=(Structured if policy=="sgr" else Normal)(journal,budget,1);planner.prompt_path=directory/"prompt.json"
            profile={**ref["profile"],"experiment_seed":seed}
            proposed=planner.propose(task,profile,[])
        finally:
            for name,value in saved.items():
                if value is None:os.environ.pop(name,None)
                else:os.environ[name]=value
        summary["calls"]+=planner.calls;summary["tokens"]+=planner.tokens
        record={"label":label,"task":task_name,"policy":policy,"seed":seed,"calls":planner.calls,
            "tokens":planner.tokens,"valid_proposal":proposed is not None,"source_reports_available":False}
        summary["arms"].append(record);save()
        if proposed is None:
            record["status"]="no_valid_proposal";save();continue
        record["original_proposal"]=proposed.to_dict();config=replace(proposed,seed=seed)
        record["executed_config"]=config.to_dict()
        main_phase="search_main" if task.classification else "search_late"
        cache={}
        try:
            case=create_trial(run/"trials",label,task.task_path,config.to_dict(),ref["sources"])
            for phase in (main_phase,"confirm","full"):
                allowance=10800-(time.monotonic()-started)
                result=run_phase(case,phase,time_limit=min(1800,allowance),trial_seconds=min(1500,allowance-10),
                    fixed_iterations=None if phase==main_phase else count,trusted_train_path=task.train_path)
                with np.load(result["prediction_path"],allow_pickle=False) as values:
                    p=(values["valid"].copy(),values["test"].copy());indices=values["valid_indices"].copy()
                expected=ref["splits"][0].valid if phase==main_phase else ref["splits"][1].valid if phase=="confirm" else np.array([],dtype=int)
                np.testing.assert_array_equal(indices,expected);cache[phase]=p
                record.setdefault("jobs",[]).append({"phase":phase,"directory":str(Path(result["phase_directory"]).relative_to(run)),
                    "seconds":result["host_elapsed_seconds"],"best_iterations":result["summary"]["best_iterations"],
                    "prediction_sha256":sha(result["prediction_path"]),"label_blinded":result["summary"]["label_blinded"]})
                if phase==main_phase:
                    count=result["summary"]["best_iterations"];yv=ref["y"][indices];base=ref["outputs"]["search"][0]
                    standalone=score(task,yv,p[0]);best=score(task,yv,base);alpha=0.
                    for weight in (.25,.5,.75,1.):
                        value=score(task,yv,(1-weight)*base+weight*p[0])
                        if improves(task,value,best):alpha,best=weight,value
                    frozen={"config":config.to_dict(),"best_iterations":count,"weight":alpha,"search_score":standalone,
                        "search_blend_score":best,"reference":ref["metadata"],"source_manifest_sha256":sha(case/"manifest.json")}
                    atomic_json(directory/"frozen_before_confirm.json",frozen);record["frozen"]=frozen
                save()
            yv=ref["y"][ref["splits"][1].valid];base=ref["outputs"]["confirm"][0]
            mixed=(1-alpha)*base+alpha*cache["confirm"][0];base_score=score(task,yv,base)
            confirm_score=score(task,yv,cache["confirm"][0]);blend_score=score(task,yv,mixed)
            selected_weight=alpha if improves(task,blend_score,base_score) else 0.
            final=(1-selected_weight)*ref["outputs"]["refit"][1]+selected_weight*cache["full"][1]
            output=directory/"submission.csv";write_submission(task,ref["test"],final,output)
            record.update(status="completed",standalone_confirm=confirm_score,frozen_blend_confirm=blend_score,
                control_confirm=base_score,selected_weight=selected_weight,
                final_confirm=blend_score if selected_weight else base_score,elapsed_seconds=time.monotonic()-started,
                submission_audit=audit_submission(task,ref["test"],output))
        except Exception as error:
            record.update(status="ml_failed",error_type=type(error).__name__,error=str(error)[-800:])
        save()
        print(json.dumps({k:record.get(k) for k in ("label","status","valid_proposal","standalone_confirm","final_confirm","tokens")},ensure_ascii=False),flush=True)
    summary["status"]="completed";save()
    print(json.dumps({"run":str(run),"calls":summary["calls"],"tokens":summary["tokens"],"arms":len(summary["arms"])},ensure_ascii=False),flush=True)


if __name__=="__main__":main()
