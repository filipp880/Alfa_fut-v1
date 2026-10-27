"""Новая проверка >=5 вариантов каждого ИИ с несколькими search и frozen confirm."""
from pathlib import Path
import os
import sys
import time
STARTED=time.monotonic()
sys.path.insert(0,str(Path(__file__).resolve().parent))
for name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"


def main(argv=None):
    import argparse
    from dataclasses import replace
    from datetime import datetime,timezone
    import json
    import threading
    import traceback
    import numpy as np
    import pandas as pd
    import psutil
    from sklearn.model_selection import StratifiedGroupKFold
    from contracts import ExperimentSpec
    from data_io import encode_target,profile_data
    from experiment_store import Journal,atomic_json
    from historical_features import HistoricalFeatures,HistorySchema
    from metrics import better,evaluate
    from proposal_catalog import catalog,parse
    from proposal_models import prior_bias,fit_sigmoid_calibration,sigmoid_calibration
    from runtime import Budget,Runner
    from submission import fallback_prediction,write_submission
    from task_reader import load_task
    from validation import Split,make_splits
    parser=argparse.ArgumentParser()
    parser.add_argument("task",nargs="?",default="task.md")
    parser.add_argument("--time-limit",type=int,default=10800)
    parser.add_argument("--trial-seconds",type=int,default=600)
    args=parser.parse_args(argv)
    if not 60<=args.time_limit<=10800:
        parser.error("Лимит задачи 60..10800 секунд")
    root=Path(args.task).resolve().parent
    work=root/".ai_proposals"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    journal=Journal(work);budget=Budget(STARTED,args.time_limit,7,24000)
    summary={"status":"initializing","experiments":[],"methods":{},"time_limit_seconds":args.time_limit,
        "network":"none","llm_calls":0,"llm_tokens":0,
        "validation_note":"Повторная оценка публичных train; прежний confirm уже исследован, это не новый слепой тест"}
    fallback_written=False
    def save(status=None):
        if status: summary["status"]=status
        summary["elapsed_seconds"]=time.monotonic()-STARTED
        atomic_json(work/"summary.json",summary)
    def stop():
        try:
            for child in reversed(psutil.Process().children(recursive=True)):
                try: child.kill()
                except psutil.Error: pass
            save("deadline")
        finally: os._exit(0 if fallback_written else 1)
    timer=threading.Timer(max(.1,budget.remaining()-.25),stop);timer.daemon=True;timer.start()
    try:
        task,train,test,_,dictionary=load_task(args.task)
        y=encode_target(task,train)
        write_submission(task,test,fallback_prediction(task,y,len(test)),root/"submission.csv")
        fallback_written=True
        full_profile,groups,times=profile_data(task,train,test,dictionary)
        standard=make_splits(task,y,groups,times)
        original_search,confirm=standard[:2]
        if task.classification:
            cv=StratifiedGroupKFold(5,shuffle=True,random_state=42)
            parts=[va for _,va in cv.split(np.arange(len(y)),y,groups)]
            other=Split("search_other",np.setdiff1d(np.arange(len(y)),np.r_[parts[1],parts[2]]),parts[2],"Второй групповой внешний search, confirm исключён")
            searches=[replace(original_search,name="search_main"),other]
            main_phase="search_main"
        else:
            levels=np.unique(times)
            early=Split("search_early",np.flatnonzero(times<levels[-3]),np.flatnonzero(times==levels[-3]),"1..2→3")
            searches=[early,replace(original_search,name="search_late")]
            main_phase="search_late"
        splits={s.name:s for s in searches+[confirm]}
        splits["full"]=Split("full",np.arange(len(y)),np.empty(0,dtype=int),"Полный refit")
        for name,split in splits.items():
            if len(split.valid): split.check(len(y),y,task.classification)
            if name.startswith("search") and np.intersect1d(split.train,confirm.valid).size:
                raise ValueError("Confirm попал в search-train")
            if task.classification and len(split.valid) and np.intersect1d(groups[split.train],groups[split.valid]).size:
                raise ValueError("Пересечение групп")
            np.savez_compressed(work/(name+".npz"),train=split.train,valid=split.valid)
        profile,_,_=profile_data(task,train.iloc[splits[main_phase].train],test,dictionary)
        profile["time_column"]=full_profile["time_column"]
        controls,specs,metadata,baseline_plan=catalog(task,dictionary,profile)
        registered={"controls":{k:v.to_dict() for k,v in controls.items()},"specs":{k:v.to_dict() for k,v in specs.items()},
            "methods":metadata,"baseline_plan":baseline_plan,"search_phases":[s.name for s in searches],"main_phase":main_phase,
            "blend_weights":[0,.25,.5,.75,1],"prior_bias_grid":[-.25,-.15,-.05,0,.05,.15,.25],
            "inner_folds":3,"seed":42,"confirm_used_for_early_stopping":False,
            "selection_rule":"Лучший pooled-search frozen blend; confirm только разрешает возврат к сильному контролю",
            "repeat_rule":"При наличии резерва один seed43 для заранее выбранного search-победителя, без изменения рецепта"}
        atomic_json(work/"preregistered.json",registered)
        summary.update(task=task.to_dict(),profile=profile,main_phase=main_phase,
            splits={name:{"train":len(s.train),"valid":len(s.valid),"reason":s.reason} for name,s in splits.items()})
        runner=Runner(work,task,profile,budget,worker_module="kaggle_worker")
        results={name:{} for name in splits};cache={};phase_predictions={};post_parameters={};oof_info={}
        print("Исследование предложений: "+str(work),flush=True)
        def run(label,spec,phase,source=None,split_path=None):
            fraction=.65 if phase.startswith("search") or phase.startswith("oof") else .85 if phase=="confirm" else .985
            allowance=min(args.trial_seconds,budget.until(fraction))
            result=runner.run(spec,split_path or work/(phase+".npz"),label+"_"+phase,allowance,
                refit=source is not None,fixed_iterations=source.get("best_iterations") if source else None)
            summary["experiments"].append(result)
            journal.event("experiment_result",recipe=label,**result)
            print(f'{label}/{phase}: {result["status"]}; {task.metric}={result.get("score")}',flush=True)
            save()
            if result["status"]=="ok":
                with np.load(Path(result["directory"])/"predictions.npz",allow_pickle=False) as values:
                    cache[result["directory"]]=(values["valid"].copy(),values["test"].copy())
            return result
        def prediction(result):
            if result.get("status")!="ok": raise RuntimeError("Нет модели: "+str(result.get("error",result.get("reason"))))
            return cache[result["directory"]]
        def mix(plan,models):
            return tuple(sum(w*prediction(models[k])[i] for w,k in zip(plan["weights"],plan["members"])) for i in (0,1))
        def save_plan(label,phase,preds,extra=None):
            path=work/f"plan_{phase}_{label}.npz"
            np.savez_compressed(path,valid=preds[0],test=preds[1],valid_indices=splits[phase].valid)
            return {"prediction_file":path.name,"score":evaluate(task,y[splits[phase].valid],preds[0])[task.metric] if len(splits[phase].valid) else None,**(extra or {})}
        def inner_oof(phase,models):
            outer=splits[phase];train_indices=outer.train
            cv=StratifiedGroupKFold(3,shuffle=True,random_state=81)
            probability=np.zeros((len(train_indices),len(task.classes)));seen=np.zeros(len(train_indices),dtype=int)
            jobs=[]
            for number,(a,b) in enumerate(cv.split(train_indices,y[train_indices],groups[train_indices])):
                tr,va=train_indices[a],train_indices[b]
                if np.intersect1d(tr,confirm.valid).size or np.intersect1d(va,confirm.valid).size or np.intersect1d(tr,outer.valid).size or np.intersect1d(va,outer.valid).size:
                    raise ValueError("Внешняя оценка попала во внутренний OOF")
                if np.intersect1d(groups[tr],groups[va]).size: raise ValueError("Утечка групп OOF")
                name=f"oof_{phase}_{number}";path=work/(name+".npz")
                np.savez_compressed(path,train=tr,valid=va)
                pieces=[]
                for member in baseline_plan["members"]:
                    result=run(member,controls[member],name,models[member],path)
                    jobs.append(result)
                    pieces.append(prediction(result)[0])
                probability[b]=sum(w*p for w,p in zip(baseline_plan["weights"],pieces));seen[b]+=1
            if not np.all(seen==1):raise ValueError("OOF не покрывает train ровно один раз")
            np.savez_compressed(work/("oof_predictions_"+phase+".npz"),prediction=probability,indices=train_indices)
            oof_info[phase]={"rows":len(train_indices),"jobs":[Path(r["directory"]).name for r in jobs],"seconds":sum(r.get("wall_seconds",0) for r in jobs)}
            return probability,y[train_indices]
        def compose(phase,models,parameters,tag=""):
            base=mix(baseline_plan,models);outputs={"baseline":base};records={"baseline":save_plan("baseline"+tag,phase,base)}
            for label,info in metadata.items():
                if label in models:
                    if models[label].get("status")!="ok":
                        records[label]={"status":models[label].get("status"),"reason":models[label].get("error",models[label].get("reason"))};continue
                    raw=prediction(models[label])
                    # Для классификационных LGB-абляций сохраняем долю прежнего CB.
                    pred=tuple(.5*raw[i]+.5*prediction(models["cb"])[i] for i in (0,1)) if task.classification else raw
                elif info.get("postprocess")=="history_gate":
                    if "history_component" not in models or models["history_component"].get("status")!="ok":
                        records[label]={"status":"skipped","reason":"Нет подходящего компонента истории"};continue
                    schema=specs["history_component"].history_schema
                    split=splits[phase]
                    hist=HistoricalFeatures(HistorySchema(**schema)).fit(train.iloc[split.train][task.feature_columns],y[split.train])
                    masks=[hist.transform(frame[task.feature_columns])["__history_known"].to_numpy(dtype=bool) for frame in (train.iloc[split.valid],test)]
                    hp=prediction(models["history_component"])
                    pred=tuple(np.where(masks[i],hp[i],base[i]) for i in (0,1))
                    np.savez_compressed(work/("gate_"+phase+tag+".npz"),valid=masks[0],test=masks[1],valid_indices=split.valid)
                elif info.get("postprocess")=="prior_bias":
                    parameter=parameters[label]
                    pred=tuple(prior_bias(p,np.asarray(parameter["prior"]),parameter["strength"]) for p in base)
                elif info.get("postprocess")=="sigmoid":
                    pred=tuple(sigmoid_calibration(p,parameters[label]["parameters"]) for p in base)
                else:continue
                outputs[label]=pred;records[label]=save_plan(label+tag,phase,pred,{"status":"ok"})
                paired=info.get("paired")
                if paired in models and models[paired].get("status")=="ok":
                    records[label]["paired_component_score"]=models[paired]["score"]
                    if label in models:records[label]["component_score"]=models[label]["score"]
            return outputs,records
        save("searching")
        for split in searches:
            phase=split.name;models=results[phase]
            for label,spec in {**controls,**specs}.items():
                models[label]=run(label,spec,phase)
            if any(models[k].get("status")!="ok" for k in baseline_plan["members"]):raise RuntimeError("Не воспроизведён сильный контроль")
            parameters={}
            if task.classification:
                probability,labels=inner_oof(phase,models)
                prior=np.bincount(labels,minlength=len(task.classes)).astype(float);prior/=prior.sum()
                ranked=[]
                for strength in registered["prior_bias_grid"]:
                    score=evaluate(task,labels,prior_bias(probability,prior,strength))[task.metric]
                    ranked.append((score,-abs(strength),strength))
                strength=max(ranked)[2]
                parameters["G5"]={"prior":prior.tolist(),"strength":strength,"oof_score":max(ranked)[0]}
                parameters["Q7"]={"parameters":fit_sigmoid_calibration(probability,labels)}
            post_parameters[phase]=parameters
            atomic_json(work/("post_parameters_"+phase+".json"),parameters)
            outputs,records=compose(phase,models,parameters)
            phase_predictions[phase]=outputs
            summary.setdefault("search_records",{})[phase]=records
            save()
        indices=np.concatenate([s.valid for s in searches]);search_y=y[indices]
        pooled={label:np.concatenate([phase_predictions[s.name][label][0] for s in searches]) for label in metadata if all(label in phase_predictions[s.name] for s in searches)}
        base_pooled=np.concatenate([phase_predictions[s.name]["baseline"][0] for s in searches])
        base_score=evaluate(task,search_y,base_pooled)[task.metric]
        frozen_methods={};winner=None;best_score=base_score
        for label,info in metadata.items():
            if label not in pooled:
                summary["methods"][label]={**info,"status":"search_failed_or_skipped"};continue
            pure=evaluate(task,search_y,pooled[label])[task.metric]
            selected_weight=0.;blend_score=base_score
            for weight in (.25,.5,.75,1.):
                score=evaluate(task,search_y,(1-weight)*base_pooled+weight*pooled[label])[task.metric]
                if better(task,score,blend_score,.001):selected_weight,blend_score=weight,score
            frozen_methods[label]={**info,"search_score":pure,"search_blend_score":blend_score,"weight":selected_weight,
                "search_by_fold":{s.name:summary["search_records"][s.name][label]["score"] for s in searches}}
            if better(task,blend_score,best_score,.001):winner,best_score=label,blend_score
        frozen={"methods":frozen_methods,"winner_by_search":winner,"baseline_search_score":base_score,
            "main_phase":main_phase,"post_parameters":post_parameters[main_phase],"oof":oof_info,
            "models":{k:{"config":v["config"],"best_iterations":v.get("best_iterations"),"status":v["status"]} for k,v in results[main_phase].items()}}
        atomic_json(work/"frozen_before_confirmation.json",frozen)
        summary.update(winner_by_search=winner,baseline_search_score=base_score,oof=oof_info)
        save("confirming")
        for label,spec in {**controls,**specs}.items():
            source=results[main_phase][label]
            if source.get("status")=="ok":results["confirm"][label]=run(label,spec,"confirm",source)
        outputs,records=compose("confirm",results["confirm"],frozen["post_parameters"])
        baseline_score=records["baseline"]["score"]
        summary["baseline"]={**records["baseline"],"plan":baseline_plan}
        for label,selected in frozen_methods.items():
            if label not in outputs:
                summary["methods"][label]={**selected,**records.get(label,{}),"status":"confirmation_failed"};continue
            weight=selected["weight"]
            blended=tuple((1-weight)*outputs["baseline"][i]+weight*outputs[label][i] for i in (0,1))
            record=save_plan(label+"_frozen_blend","confirm",blended)
            summary["methods"][label]={**selected,"status":"ok","confirmation_score":records[label]["score"],
                "confirmation_prediction_file":records[label]["prediction_file"],"confirmation_blend_score":record["score"],
                "blend_prediction_file":record["prediction_file"],"paired_component_score":records[label].get("paired_component_score"),
                "component_score":records[label].get("component_score")}
        use_new=winner is not None and summary["methods"].get(winner,{}).get("status")=="ok" and better(task,summary["methods"][winner]["confirmation_blend_score"],baseline_score,.001)
        selected=winner if use_new else "baseline"
        weight=frozen_methods[selected]["weight"] if use_new else 0.
        predicted=outputs["baseline"][1] if selected=="baseline" else (1-weight)*outputs["baseline"][1]+weight*outputs[selected][1]
        write_submission(task,test,predicted,root/"submission.csv")
        summary.update(selected=selected,selected_weight=weight,confirmation_records=records)
        # Заранее назначенный повтор search-победителя: фиксированные параметры, seed43.
        if winner and budget.remaining()>300:
            save("repeating")
            repeat_models={}
            needed=set(baseline_plan["members"])
            if winner in specs:needed.add(winner)
            if winner=="G6":needed.add("history_component")
            for label in sorted(needed):
                original=results[main_phase][label]
                data=dict(original["config"]);data["seed"]=43
                if "base_config" in data:data["base_config"]={**data["base_config"],"seed":43}
                repeat_models[label]=run("repeat43_"+label,parse(data,task),"confirm",original)
            if all(r.get("status")=="ok" for r in repeat_models.values()):
                # compose пропускает необученные обычные кандидаты, постпроцессоры используют frozen параметры.
                repeat_outputs,repeat_records=compose("confirm",repeat_models,frozen["post_parameters"],tag="_repeat43")
                if winner in repeat_outputs:
                    b=repeat_outputs["baseline"];candidate=repeat_outputs[winner];alpha=frozen_methods[winner]["weight"]
                    p=(1-alpha)*b[0]+alpha*candidate[0]
                    np.savez_compressed(work/"repeat43_predictions.npz",baseline=b[0],candidate=candidate[0],blend=p,valid_indices=confirm.valid)
                    summary["repeat43"]={"winner":winner,"baseline_score":evaluate(task,y[confirm.valid],b[0])[task.metric],
                        "pure_score":evaluate(task,y[confirm.valid],candidate[0])[task.metric],"blend_score":evaluate(task,y[confirm.valid],p)[task.metric]}
        save("refitting")
        needed=set(baseline_plan["members"])
        if selected in specs:needed.add(selected)
        if selected=="G6":needed.add("history_component")
        for label in sorted(needed):
            results["full"][label]=run(label,{**controls,**specs}[label],"full",results[main_phase][label])
        full_ok=all(results["full"][k].get("status")=="ok" for k in needed)
        if full_ok:
            full_outputs,_=compose("full",results["full"],frozen["post_parameters"])
            final=full_outputs["baseline"][1] if selected=="baseline" else (1-weight)*full_outputs["baseline"][1]+weight*full_outputs[selected][1]
            write_submission(task,test,final,root/"submission.csv")
        summary.update(final_refit_completed=full_ok,peak_worker_rss_mb=max(r.get("peak_memory_mb",0) for r in summary["experiments"]))
        save("completed")
        return 0
    except Exception as error:
        summary["error"]=str(error)[-1800:];traceback.print_exc();save("failed");return 1
    finally:timer.cancel()


if __name__=="__main__":
    raise SystemExit(main())
