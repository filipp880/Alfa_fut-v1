"""Независимый пересчёт предложений GPT/Qwen из сохранённых NPZ и запросов."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
for name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[name]="1"
import numpy as np
import pandas as pd
from contracts import native
from data_io import encode_target,feature_groups,profile_data
from metrics import evaluate,better
from proposal_catalog import parse
from proposal_models import prior_bias,fit_sigmoid_calibration,sigmoid_calibration
from submission import make_submission
from report_unconventional import paired_interval
from task_reader import load_task
from validation import make_splits


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def base_name(path):
    return str(path).replace("\\","/").rsplit("/",1)[-1]


def check(actual,expected,label):
    if expected is None or not np.isclose(actual,expected,atol=1e-10,rtol=1e-10):
        raise ValueError(f"Метрика не воспроизводится: {label}: {actual} != {expected}")


def load(path,indices,test_rows):
    with np.load(path,allow_pickle=False) as values:
        np.testing.assert_array_equal(values["valid_indices"],indices)
        valid,test=values["valid"].copy(),values["test"].copy()
        if len(valid)!=len(indices) or len(test)!=test_rows or not np.isfinite(valid).all() or not np.isfinite(test).all():
            raise ValueError("Некорректные предсказания: "+str(path))
        return valid,test


def analyze(case):
    from sklearn.model_selection import StratifiedGroupKFold
    case=Path(case).resolve()
    paths=sorted(case.glob(".ai_proposals/*/summary.json"))
    if not paths:raise ValueError("Нет результатов: "+str(case))
    work=paths[-1].parent;s=read(paths[-1])
    if s.get("status")!="completed":raise ValueError("Исследование не завершено: "+str(s.get("status")))
    registered=read(work/"preregistered.json");frozen_path=work/"frozen_before_confirmation.json";frozen=read(frozen_path)
    task,train,test,_,dictionary=load_task(case/"task.md")
    y=encode_target(task,train);groups=feature_groups(task,train)
    _,_,times=profile_data(task,train,test,dictionary)
    originals=make_splits(task,y,groups,times);confirm=originals[1]
    split_cache={}
    for path in work.glob("*.npz"):
        with np.load(path,allow_pickle=False) as values:
            if set(values.files)=={"train","valid"}:
                split_cache[path.stem]=(values["train"].copy(),values["valid"].copy())
    np.testing.assert_array_equal(split_cache["confirm"][0],confirm.train)
    np.testing.assert_array_equal(split_cache["confirm"][1],confirm.valid)
    if task.classification:
        cv=StratifiedGroupKFold(5,shuffle=True,random_state=42)
        parts=[va for _,va in cv.split(np.arange(len(y)),y,groups)]
        expected={"search_main":(originals[0].train,originals[0].valid),
                  "search_other":(np.setdiff1d(np.arange(len(y)),np.r_[parts[1],parts[2]]),parts[2])}
    else:
        level=np.unique(times)[-3]
        expected={"search_early":(np.flatnonzero(times<level),np.flatnonzero(times==level)),
                  "search_late":(originals[0].train,originals[0].valid)}
    for phase,pair in expected.items():
        for a,b in zip(split_cache[phase],pair):np.testing.assert_array_equal(a,b)
    models={};jobs=[];results_by_directory={}
    for path in sorted(work.glob("*/request.json")):
        directory=path.parent;request=read(path);phase=base_name(request["split_path"]).removesuffix(".npz")
        tr,va=split_cache[phase]
        if np.intersect1d(tr,va).size:raise ValueError("Пересечение train/valid")
        if phase.startswith("search") or phase.startswith("oof"):
            if np.intersect1d(tr,confirm.valid).size or np.intersect1d(va,confirm.valid).size:
                raise ValueError("Confirm попал в поиск/OOF")
            if (work/"preregistered.json").stat().st_mtime_ns>path.stat().st_mtime_ns:
                raise ValueError("Протокол записан после fit")
        if task.classification and len(va) and np.intersect1d(groups[tr],groups[va]).size:
            raise ValueError("Пересечение групп")
        if times is not None and len(va) and np.max(times[tr])>=np.min(times[va]):raise ValueError("Нарушена хронология")
        result_path=directory/"controller_result.json"
        result=read(result_path) if result_path.exists() else {"status":"skipped","reason":"Нет controller_result"}
        label=result.get("phase",directory.name).removesuffix("_"+phase)
        spec=parse(request["config"],task)
        item={"directory":directory.name,"label":label,"phase":phase,"status":result["status"],
            "config_key":spec.key,"wall_seconds":result.get("wall_seconds",0),"peak_memory_mb":result.get("peak_memory_mb",0),
            "best_iterations":result.get("best_iterations"),"reason":result.get("error",result.get("reason")),"diagnostics":result.get("diagnostics")}
        if phase in {"confirm","full"}:
            if not request.get("refit") or frozen_path.stat().st_mtime_ns>path.stat().st_mtime_ns:
                raise ValueError("Подтверждение без фиксации/с early stop")
            original_label=label.removeprefix("repeat43_")
            reference=frozen["models"][original_label]
            expected_config=dict(reference["config"])
            if label.startswith("repeat43_"):
                expected_config["seed"]=43
                if "base_config" in expected_config:expected_config["base_config"]={**expected_config["base_config"],"seed":43}
            if request["config"]!=expected_config or request.get("fixed_iterations")!=reference["best_iterations"]:
                raise ValueError("Изменены конфигурация/итерации после search")
        if result["status"]=="ok":
            predictions=load(directory/"predictions.npz",va,len(test))
            if result["n_train"]!=len(tr):raise ValueError("Неверный размер обучения")
            if len(va):
                score=evaluate(task,y[va],predictions[0])[task.metric];check(score,result.get("score"),directory.name);item["score"]=score
            if phase in {"confirm","full"} and result["best_iterations"]!=request["fixed_iterations"]:
                raise ValueError("Выполнено неверное число фиксированных итераций")
            if phase in {"confirm","full"} and result.get("evaluation_used_for_fit",False):raise ValueError("Confirm использован для fit")
            models[(phase,label)]={"predictions":predictions,"request":request,"result":result,"directory":directory.name}
        jobs.append(item);results_by_directory[directory.name]=result
    baseline_plan=registered["baseline_plan"]
    def compose_baseline(phase,prefix=""):
        return tuple(sum(w*models[(phase,prefix+k)]["predictions"][i] for w,k in zip(baseline_plan["weights"],baseline_plan["members"])) for i in (0,1))
    phase_predictions={}
    def compose_phase(phase,prefix="",tag=""):
        base=compose_baseline(phase,prefix);values={"baseline":base}
        parameters=frozen["post_parameters"] if phase in {"confirm","full"} else read(work/("post_parameters_"+phase+".json")) if (work/("post_parameters_"+phase+".json")).exists() else {}
        for label,info in registered["methods"].items():
            if (phase,prefix+label) in models:
                raw=models[(phase,prefix+label)]["predictions"]
                values[label]=tuple(.5*raw[i]+.5*models[(phase,prefix+"cb")]["predictions"][i] for i in (0,1)) if task.classification else raw
            elif info.get("postprocess")=="history_gate" and (phase,prefix+"history_component") in models:
                from historical_features import HistoricalFeatures,HistorySchema
                tr,va=split_cache[phase]
                schema=registered["specs"]["history_component"]["history_schema"]
                history=HistoricalFeatures(HistorySchema(**schema)).fit(train.iloc[tr][task.feature_columns],y[tr])
                masks=[history.transform(frame[task.feature_columns])["__history_known"].to_numpy(dtype=bool) for frame in (train.iloc[va],test)]
                with np.load(work/("gate_"+phase+tag+".npz")) as saved:
                    for k,mask in zip(("valid","test"),masks):np.testing.assert_array_equal(mask,saved[k])
                raw=models[(phase,prefix+"history_component")]["predictions"]
                values[label]=tuple(np.where(masks[i],raw[i],base[i]) for i in (0,1))
            elif info.get("postprocess")=="prior_bias":
                p=parameters[label];values[label]=tuple(prior_bias(a,np.asarray(p["prior"]),p["strength"]) for a in base)
            elif info.get("postprocess")=="sigmoid":
                values[label]=tuple(sigmoid_calibration(a,parameters[label]["parameters"]) for a in base)
        for label,predictions in values.items():
            saved=load(work/f"plan_{phase}_{label}{tag}.npz",split_cache[phase][1],len(test))
            for a,b in zip(predictions,saved):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
        return values
    for phase in registered["search_phases"]+["confirm"]:
        phase_predictions[phase]=compose_phase(phase)
    indices=np.concatenate([split_cache[p][1] for p in registered["search_phases"]])
    target=y[indices]
    baseline_search=np.concatenate([phase_predictions[p]["baseline"][0] for p in registered["search_phases"]])
    baseline_confirm=phase_predictions["confirm"]["baseline"][0]
    base_score=evaluate(task,y[confirm.valid],baseline_confirm)[task.metric]
    check(base_score,s["baseline"]["score"],"baseline confirm")
    base_search_score=evaluate(task,target,baseline_search)[task.metric]
    check(base_search_score,frozen["baseline_search_score"],"baseline search")
    # OOF-предсказания должны быть собраны ровно из внутренних, непересекающихся fit.
    for phase,info in frozen.get("oof",{}).items():
        outer_train=split_cache[phase][0]
        with np.load(work/("oof_predictions_"+phase+".npz")) as data:
            np.testing.assert_array_equal(data["indices"],outer_train);saved=data["prediction"]
        positions={int(index):i for i,index in enumerate(outer_train)}
        assembled=np.zeros_like(saved);seen=np.zeros(len(outer_train),dtype=int)
        for n in range(3):
            inner=f"oof_{phase}_{n}";tr,va=split_cache[inner]
            if not set(np.r_[tr,va])<=set(outer_train):raise ValueError("OOF выходит за внешний train")
            prediction=compose_baseline(inner)[0];where=np.array([positions[int(i)] for i in va])
            assembled[where]=prediction;seen[where]+=1
            for member in baseline_plan["members"]:
                request=models[(inner,member)]["request"]
                if not request["refit"] or request["fixed_iterations"]!=models[(phase,member)]["result"]["best_iterations"]:
                    raise ValueError("Не зафиксированы OOF-итерации")
        np.testing.assert_array_equal(seen,np.ones(len(outer_train),dtype=int));np.testing.assert_allclose(assembled,saved,rtol=1e-12,atol=1e-12)
        parameters=read(work/("post_parameters_"+phase+".json"))
        prior=np.bincount(y[outer_train],minlength=len(task.classes)).astype(float);prior/=prior.sum()
        ranked=[(evaluate(task,y[outer_train],prior_bias(saved,prior,strength))[task.metric],-abs(strength),strength)
                for strength in registered["prior_bias_grid"]]
        if parameters["G5"]["strength"]!=max(ranked)[2]:raise ValueError("Prior bias выбран не по внутреннему OOF")
        np.testing.assert_allclose(parameters["G5"]["prior"],prior,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(parameters["Q7"]["parameters"],fit_sigmoid_calibration(saved,y[outer_train]),rtol=1e-6,atol=1e-6)
        if phase==frozen["main_phase"] and parameters!=frozen["post_parameters"]:raise ValueError("Параметры коррекции изменены после search")
    sign=1 if task.maximize else -1
    methods={};best_label=None;best_score=base_search_score
    for label,record in s["methods"].items():
        if record.get("status")!="ok":methods[label]=record;continue
        frozen_record=frozen["methods"][label]
        if frozen_record["weight"]!=record["weight"]:raise ValueError("Вес поменялся после confirm")
        pooled=np.concatenate([phase_predictions[p][label][0] for p in registered["search_phases"]])
        check(evaluate(task,target,pooled)[task.metric],record["search_score"],label+" pooled")
        search_blend=(1-record["weight"])*baseline_search+record["weight"]*pooled
        check(evaluate(task,target,search_blend)[task.metric],record["search_blend_score"],label+" search blend")
        replay_weight=0.;replay_score=base_search_score
        for alpha in registered["blend_weights"][1:]:
            value=evaluate(task,target,(1-alpha)*baseline_search+alpha*pooled)[task.metric]
            if better(task,value,replay_score,.001):replay_weight,replay_score=alpha,value
        if replay_weight!=record["weight"]:raise ValueError("Вес выбран не по зарегистрированной сетке search")
        for phase,expected_score in record["search_by_fold"].items():check(evaluate(task,y[split_cache[phase][1]],phase_predictions[phase][label][0])[task.metric],expected_score,label+phase)
        weight=record["weight"];candidate=phase_predictions["confirm"][label][0]
        blended=(1-weight)*baseline_confirm+weight*candidate
        pure_metrics=evaluate(task,y[confirm.valid],candidate);blend_metrics=evaluate(task,y[confirm.valid],blended)
        check(pure_metrics[task.metric],record["confirmation_score"],label+" confirm")
        check(blend_metrics[task.metric],record["confirmation_blend_score"],label+" blend")
        saved=load(work/record["blend_prediction_file"],confirm.valid,len(test))[0];np.testing.assert_allclose(saved,blended,rtol=1e-12,atol=1e-12)
        if better(task,record["search_blend_score"],best_score,.001):best_label,best_score=label,record["search_blend_score"]
        method_labels={label}
        if label=="G6":method_labels={"history_component"}
        own_jobs=[j for j in jobs if j["label"] in method_labels and not j["phase"].startswith("oof") and j["phase"]!="full"]
        oof_seconds=sum(v["seconds"] for v in frozen.get("oof",{}).values()) if label in {"G5","Q7"} else 0.
        delta=sign*(pure_metrics[task.metric]-base_score)
        methods[label]={**record,"metrics":pure_metrics,"blend_metrics":blend_metrics,
            "absolute_improvement":delta,"relative_improvement_percent":delta/abs(base_score)*100,
            "blend_absolute_improvement":sign*(blend_metrics[task.metric]-base_score),
            "improvement_ci95_conditional":paired_interval(task,y[confirm.valid],baseline_confirm,candidate,groups[confirm.valid],400),
            "blend_improvement_ci95_conditional":([0.,0.] if weight==0 else paired_interval(task,y[confirm.valid],baseline_confirm,blended,groups[confirm.valid],400)),
            "fit_seconds":sum(j["wall_seconds"] for j in own_jobs),"shared_oof_fit_seconds":oof_seconds,
            "peak_memory_mb":max((j["peak_memory_mb"] for j in own_jobs),default=0),
            "diagnostics":[j["diagnostics"] for j in own_jobs if j["diagnostics"]]}
    if best_label!=s["winner_by_search"]:raise ValueError("Выбор search-победителя не воспроизводится")
    if s.get("repeat43"):
        outputs=compose_phase("confirm",prefix="repeat43_",tag="_repeat43")
        label=s["repeat43"]["winner"];weight=frozen["methods"][label]["weight"]
        b,c=outputs["baseline"][0],outputs[label][0];blend=(1-weight)*b+weight*c
        for name,prediction in (("baseline_score",b),("pure_score",c),("blend_score",blend)):
            check(evaluate(task,y[confirm.valid],prediction)[task.metric],s["repeat43"][name],"repeat43 "+name)
    expected_new=best_label is not None and better(task,methods[best_label]["confirmation_blend_score"],base_score,.001)
    if s["selected"]!=(best_label if expected_new else "baseline"):raise ValueError("Резервный выбор не воспроизводится")
    if not s["final_refit_completed"]:raise ValueError("Не завершён полный refit")
    full_outputs=compose_phase("full")
    final=full_outputs["baseline"][1]
    if s["selected"]!="baseline":
        alpha=frozen["methods"][s["selected"]]["weight"]
        final=(1-alpha)*final+alpha*full_outputs[s["selected"]][1]
    expected_csv=make_submission(task,test,final)
    actual_csv=pd.read_csv(case/"submission.csv",dtype={c:str for c in task.id_columns},keep_default_na=False)
    pd.testing.assert_frame_equal(actual_csv,expected_csv,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    segments={}
    if "G6" in methods:
        with np.load(work/"gate_confirm.npz") as data:known=data["valid"].astype(bool)
        for name,mask in (("known",known),("unknown",~known)):
            segments[name]={"rows":int(mask.sum()),"fraction":float(mask.mean()),
                "baseline_rmse":evaluate(task,y[confirm.valid][mask],baseline_confirm[mask])[task.metric],
                "history_rmse":evaluate(task,y[confirm.valid][mask],models[("confirm","history_component")]["predictions"][0][mask])[task.metric],
                "gate_rmse":evaluate(task,y[confirm.valid][mask],phase_predictions["confirm"]["G6"][0][mask])[task.metric]}
        with np.load(work/"gate_full.npz") as data:segments["test_known_fraction"]=float(data["test"].mean())
    manifest=read(case/"docker_objects.json");audit=read(case/"resource_audit.json");deps=read(case/"dependency_audit.json")
    if not(audit["cpu_total"]<=8 and audit["memory_bytes_total"]<=32000000000
           and audit["memory_swap_bytes"]==audit["memory_bytes_total"] and audit["readonly_root"]
           and audit["network"]=="none" and not deps["mismatches"]):raise ValueError("Не соблюдены условия стенда")
    if deps["requirements_sha256"]!="9173b0388532fbd2a709e4f4b3029e3b9ccab685fce43a85261ffa1ba95efbe6":raise ValueError("Изменены требования")
    archive=case/"source.zip"
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest["archive_sha256"]:raise ValueError("SHA ZIP не совпадает")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() or any("/" in n or "\\" in n for n in bundle.namelist()):raise ValueError("Неверный ZIP")
        for name in bundle.namelist():
            if bundle.read(name)!=(case/name).read_bytes():raise ValueError("Snapshot изменён: "+name)
    if s["elapsed_seconds"]>10800 or s["llm_calls"] or s["llm_tokens"]:raise ValueError("Нарушен бюджет исследования")
    selected_score=base_score if s["selected"]=="baseline" else methods[s["selected"]]["confirmation_blend_score"]
    return {"task":task.problem_type,"metric":task.metric,"directory":str(case.relative_to(ROOT)),"summary_directory":work.name,
        "baseline_score":base_score,"baseline_search_score":base_search_score,"baseline_metrics":evaluate(task,y[confirm.valid],baseline_confirm),
        "baseline_by_fold":{p:evaluate(task,y[split_cache[p][1]],phase_predictions[p]["baseline"][0])[task.metric] for p in registered["search_phases"]},
        "methods":methods,"winner_by_search":s["winner_by_search"],"selected":s["selected"],"selected_score":selected_score,
        "repeat43":s.get("repeat43"),"segments":segments,"elapsed_seconds":s["elapsed_seconds"],"experiment_count":len(jobs),"experiments":jobs,
        "peak_worker_rss_mb":s["peak_worker_rss_mb"],"final_refit_completed":s["final_refit_completed"],"sources":manifest["sources"],
        "resource_audit":audit,"dependency_audit":deps,"archive_sha256":manifest["archive_sha256"],
        "checks":{"metrics_recomputed":True,"exact_indices":True,"search_excludes_confirm":True,"frozen_before_confirm":True,
                  "fixed_iterations":True,"inner_oof_checked":True,"source_zip_intact":True,"network_none":True,"final_csv_matches_full_refit":True},
        "interpretation":"Повторная исследовательская оценка; условные интервалы не учитывают выбор по известным данным и будущий сдвиг."}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("cases",nargs="+");parser.add_argument("--output",default=str(ROOT/"docs"/"AI_PROPOSALS_RESULTS.json"))
    args=parser.parse_args();results=[analyze(case) for case in args.cases]
    output=Path(args.output);output.write_text(json.dumps(native(results),ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    for result in results:
        print(f'{result["task"]}: контроль {result["baseline_score"]:.6f}, выбран {result["selected"]}, метрика {result["selected_score"]:.6f}')
    print("Независимо проверенный отчёт: "+str(output))


if __name__=="__main__":main()
