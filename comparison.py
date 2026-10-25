"""Сравнение трёх стратегий в одном запуске задачи, строго на данных кейса.

Исследовательский entrypoint упаковывается под именем solution.py.
"""
from __future__ import annotations
import os
from pathlib import Path
import sys
import time

STARTED = time.monotonic()
sys.path.insert(0,str(Path(__file__).resolve().parent))
for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"


def main(argv=None):
    import argparse
    from dataclasses import replace
    from datetime import datetime,timezone
    import importlib.metadata
    import json
    import threading
    import traceback
    import numpy as np
    import optuna
    import psutil
    from contracts import ExperimentSpec
    from data_io import encode_target,profile_data
    from task_reader import load_task
    from validation import make_splits
    from submission import fallback_prediction,write_submission
    from experiment_store import Journal,atomic_json
    from llm_client import LLMPlanner
    from research import Research
    from runtime import Budget,Runner
    from metrics import better
    from strategy_specs import AutoGluonSpec,validated_space,validate_semantic_transforms

    parser = argparse.ArgumentParser()
    parser.add_argument("task",nargs="?",default="task.md")
    parser.add_argument("--time-limit",type=float,default=10800)
    parser.add_argument("--control-trials",type=int,default=8)
    parser.add_argument("--optuna-trials",type=int,default=12)
    parser.add_argument("--feature-rounds",type=int,default=2)
    parser.add_argument("--max-iterations",type=int,default=2400)
    parser.add_argument("--trial-seconds",type=float,default=200)
    parser.add_argument("--branch-seconds",type=float,default=1800)
    parser.add_argument("--automl-seconds",type=float,default=1800)
    parser.add_argument("--no-llm",action="store_true")
    args = parser.parse_args(argv)
    if not 30<=args.time_limit<=10800:
        parser.error("Лимит одной задачи: 30..10800 секунд")
    budget = Budget(STARTED,args.time_limit,7,24000)
    root = Path(args.task).resolve().parent
    work = root/".comparison"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")+f"_{os.getpid()}")
    journal = Journal(work)
    summary = {"status":"initializing","methods":{},"time_limit_seconds":budget.seconds,
               "threads":budget.threads,"validation_policy":"Подтверждение без ранней остановки и подбора"}
    fallback_written = False

    def save(status=None):
        if status:
            summary["status"] = status
        summary["elapsed_seconds"] = time.monotonic()-STARTED
        atomic_json(work/"comparison_summary.json",summary)

    def stop():
        try:
            for child in reversed(psutil.Process().children(recursive=True)):
                try: child.kill()
                except psutil.Error: pass
            save("deadline")
        finally:
            os._exit(0 if fallback_written else 1)

    timer = threading.Timer(max(.1,budget.remaining()-.25),stop)
    timer.daemon=True
    timer.start()
    try:
        task,train,test,_,dictionary = load_task(args.task)
        y = encode_target(task,train)
        write_submission(task,test,fallback_prediction(task,y,len(test)),root/"submission.csv")
        fallback_written=True
        if os.environ.get("ALFA_RESTRICTED_NETWORK")=="1":
            import httpx
            blocked=False
            try:
                httpx.get("https://forbidden.invalid/",timeout=5)
            except httpx.ProxyError:
                blocked=True
            if not blocked:
                raise RuntimeError("Не подтверждена блокировка посторонних узлов")
            network_audit={"unapproved_connect_blocked":True,"allowed_host":"api.alfahack-gateway26.ru"}
            if not args.no_llm:
                response=httpx.get("https://api.alfahack-gateway26.ru/v1/models",timeout=15,
                                   headers={"Authorization":"Bearer "+os.environ["OPENAI_API_KEY"]})
                response.raise_for_status()
                network_audit["model_ids"]=[item["id"] for item in response.json().get("data",[])]
            atomic_json(work/"network_audit.json",network_audit)
        full_profile,groups,times = profile_data(task,train,test,dictionary)
        splits = make_splits(task,y,groups,times)
        if len(splits)<2:
            raise ValueError("Для независимого сравнения нужна отдельная подтверждающая часть")
        search,confirm = splits[:2]
        if set(np.r_[search.train,search.valid]) != set(confirm.train):
            raise ValueError("Поиск должен покрывать ровно train-часть подтверждения")
        profile,_,_ = profile_data(task,train.iloc[search.train],test,dictionary)
        profile["time_column"] = full_profile["time_column"]
        for split in splits:
            np.savez_compressed(work/f"split_{split.name}.npz",train=split.train,valid=split.valid)
        np.savez_compressed(work/"split_full.npz",train=np.arange(len(train)),valid=np.empty(0,dtype=int))
        versions = {}
        for package in ("numpy","scipy","pandas","scikit-learn","catboost","lightgbm","xgboost","optuna","autogluon.tabular","PyYAML"):
            versions[package] = importlib.metadata.version(package)
        summary.update(task={"target":task.target,"problem_type":task.problem_type,"metric":task.metric,
                            "train_rows":len(train),"test_rows":len(test)},
                       versions=versions,python=sys.version,profile=profile,
                       split_sizes={s.name:{"train":len(s.train),"valid":len(s.valid),"reason":s.reason} for s in splits})
        save("searching")
        planner = LLMPlanner(journal,budget,0 if args.no_llm else 3)
        runner = Runner(work,task,profile,budget,worker_module="strategy_worker")
        primary_cache,confirm_cache = {},{}
        all_results=[]
        best_output_score = None

        def run(config,method,phase="search",timeout=None,source=None):
            nonlocal best_output_score
            config.validate(task)
            cache = primary_cache if phase=="search" else confirm_cache if phase=="confirm" else {}
            key=config.key
            if key in cache:
                journal.event("experiment_reused",phase=method+"_"+phase,config_key=key)
                return cache[key]
            options={}
            if source is not None:
                options={"source_directory":source["directory"],"use_all_training":phase=="full"}
            deadline = budget.until(.73) if phase=="search" else budget.until(.91) if phase=="confirm" else budget.until(.985)
            allowance = min(timeout or args.trial_seconds,deadline)
            if allowance<5:
                return {"status":"skipped","reason":"Резерв завершения задачи","config":config.to_dict()}
            fixed = source["best_iterations"] if source and config.model!="autogluon" else None
            journal.event("experiment_started",phase=method+"_"+phase,model=config.model,hypothesis=config.hypothesis)
            result = runner.run(config,work/f"split_{phase if phase!='full' else 'full'}.npz",
                                method+"_"+phase,allowance,refit=phase!="search",
                                fixed_iterations=fixed,job_options=options)
            result["method"] = method
            all_results.append(result)
            journal.event("experiment_result",**result)
            if result.get("status")=="ok":
                cache[key]=result
                if phase=="search" and better(task,result["score"],best_output_score):
                    with np.load(Path(result["directory"])/"predictions.npz",allow_pickle=False) as values:
                        write_submission(task,test,values["test"],root/"submission.csv")
                    best_output_score=result["score"]
            return result

        def ranked(results):
            return sorted([r for r in results if r.get("status")=="ok"],
                          key=lambda r:r["score"],reverse=task.maximize)

        offline = type("Offline",(),{"available":False})()
        research = Research(task,profile,offline)
        controls=[]
        for _ in range(args.control_trials):
            config=research.next(controls)
            if config is None: break
            config=replace(config,iterations=min(config.iterations,args.max_iterations))
            controls.append(run(config,"control"))
        if not ranked(controls):
            raise RuntimeError("Контрольный ML-движок не получил валидную модель")
        control=ranked(controls)[0]
        baseline_config=ExperimentSpec.from_dict(control["config"],task)
        candidates={"control":control}

        # 1. Optuna + однократное LLM-ограничение осмысленного пространства.
        proposal=planner.request(
            "Предложи диапазоны гиперпараметров для CatBoost/LightGBM. Верни JSON с ключами "
            "depth,learning_rate,regularization,leaves,min_child_samples и массивами [min,max]. "
            "Не меняй данные, метрику и валидацию. Сохраняй возможности около лучших моделей.",
            {"metric":task.metric,"profile":profile,"best_configs":[r["config"] for r in ranked(controls)[:2]]})
        space=validated_space(proposal)
        for name in ("depth","learning_rate","regularization","leaves","min_child_samples"):
            space[name][0]=min(space[name][0],getattr(baseline_config,name))
            space[name][1]=max(space[name][1],getattr(baseline_config,name))
        atomic_json(work/"optuna_space.json",space)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study=optuna.create_study(direction="maximize" if task.maximize else "minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=4))
        iteration_choices=sorted({min(args.max_iterations,n) for n in (800,1200,1600,2400,3200)})
        weight_choices=[0.,.25,.5,.75,1.] if task.classification else [0.]

        def to_params(config):
            result={"model":config.model,"iterations":config.iterations,"learning_rate":config.learning_rate,
                    "regularization":config.regularization,"class_weight_power":config.class_weight_power,
                    "missing_indicators":config.missing_indicators,"frequency_encoding":config.frequency_encoding,
                    "drop_time":config.drop_time,"target_transform":config.target_transform}
            if config.model=="catboost": result["depth"]=config.depth
            else: result.update(leaves=config.leaves,min_child_samples=config.min_child_samples)
            return result

        for previous in ranked(controls)[:2]:
            study.enqueue_trial(to_params(ExperimentSpec.from_dict(previous["config"],task)))
        hpo=[control]
        branch_end=time.monotonic()+args.branch_seconds
        for _ in range(args.optuna_trials):
            if time.monotonic()>=branch_end or budget.until(.73)<8: break
            trial=study.ask()
            model=trial.suggest_categorical("model",["catboost","lightgbm"])
            parameters={"model":model,"iterations":trial.suggest_categorical("iterations",iteration_choices),
                "learning_rate":trial.suggest_float("learning_rate",*space["learning_rate"],log=True),
                "regularization":trial.suggest_float("regularization",*space["regularization"],log=True),
                "class_weight_power":trial.suggest_categorical("class_weight_power",weight_choices),
                "missing_indicators":trial.suggest_categorical("missing_indicators",[False,True]) if profile["missing_train"] else False,
                "frequency_encoding":trial.suggest_categorical("frequency_encoding",[False,True]) if task.categorical_columns else False,
                "drop_time":trial.suggest_categorical("drop_time",[False,True]) if profile["time_column"] else False,
                "target_transform":trial.suggest_categorical("target_transform",["identity","log1p"])
                    if not task.classification and profile.get("target_summary",{}).get("nonnegative") else "identity",
                "hypothesis":"Optuna TPE: исследование параметров в проверенных границах"}
            if model=="catboost": parameters["depth"]=trial.suggest_int("depth",*space["depth"])
            else:
                leaves=[v for v in (15,31,63,127) if space["leaves"][0]<=v<=space["leaves"][1]]
                parameters["leaves"]=trial.suggest_categorical("leaves",leaves or [31])
                parameters["min_child_samples"]=trial.suggest_int("min_child_samples",*space["min_child_samples"])
            config=ExperimentSpec.from_dict(parameters,task)
            result=run(config,"optuna",timeout=min(args.trial_seconds,branch_end-time.monotonic()))
            if result.get("status")=="ok":
                study.tell(trial,result["score"])
                hpo.append(result)
            else: study.tell(trial,state=optuna.trial.TrialState.FAIL)
        atomic_json(work/"optuna_trials.json",{"trials":[{"number":t.number,"params":t.params,"value":t.value,"state":t.state.name}
                                                          for t in study.trials]})
        candidates["optuna"]=ranked(hpo)[0]

        # 2. Признаки от Qwen: единицы и расчёты проверяются до обучения.
        features=[control]
        metadata=[]
        if not dictionary.empty:
            metadata=[{k:str(row.get(k,""))[:130] for k in ("feature","dtype","unit","description_ru")}
                      for row in dictionary.fillna("").to_dict("records") if row.get("feature") in task.feature_columns]
        descriptions=[]
        for round_number in range(args.feature_rounds):
            response=planner.request(
                "Ты исследователь признаков табличной ML-задачи. Верни JSON {\"hypotheses\":["
                "{\"hypothesis\":\"обоснование по-русски\",\"transforms\":[{\"op\":\"ratio\",\"columns\":[\"a\",\"b\"]}]}]}. "
                "Предложи 3 разные группы по 1-3 признака. op: ratio,difference,sum,log1p,missing. "
                "Для log1p/missing один столбец, для остальных два. Используй только существующие признаки, "
                "их смысл и единицы. Таргет и id запрещены. Не дублируй готовые показатели. "
                "Нельзя вычитать анонимизированный код периода из календарного года. Не пиши код.",
                {"task_excerpt":Path(task.task_path).read_text(encoding="utf-8-sig")[:1800],
                 "metric":task.metric,"dictionary":metadata,"profile":profile,
                 "baseline_metrics":control["metrics"],"previous_hypotheses":descriptions[-6:]})
            hypotheses=response.get("hypotheses",[]) if isinstance(response,dict) else []
            if not isinstance(hypotheses,list): hypotheses=[]
            for hypothesis in hypotheses[:3]:
                try:
                    if not isinstance(hypothesis,dict): raise ValueError("Неверная схема гипотезы")
                    transforms=validate_semantic_transforms(task,train.iloc[search.train],dictionary,hypothesis.get("transforms",[]))
                    if not transforms: raise ValueError("Нет новых допустимых признаков")
                    config=replace(baseline_config,transforms=transforms,
                                   hypothesis=str(hypothesis.get("hypothesis","Семантические признаки"))[:1000]).validate(task)
                    result=run(config,"semantic")
                    descriptions.append({"hypothesis":config.hypothesis,"transforms":transforms,
                                         "score":result.get("score"),"status":result.get("status")})
                    features.append(result)
                except (ValueError,TypeError,KeyError) as error:
                    journal.event("feature_hypothesis_rejected",reason=str(error)[:500])
        atomic_json(work/"semantic_hypotheses.json",{"tested":descriptions})
        candidates["semantic"]=ranked(features)[0]

        # 3. AutoGluon: только GBM/CAT/XGB; групповые bag-folds либо без bagging для времени.
        ag=AutoGluonSpec(iterations=args.max_iterations,class_weight_power=.5 if task.classification
                        and task.metric in {"macro_f1","f1","balanced_accuracy"} else 0,
                        bag_folds=0 if profile["time_column"] else 3,
                        hypothesis="AutoGluon: разрешённые бустинги с корректной валидацией")
        ag_result=run(ag,"automl",timeout=min(args.automl_seconds,budget.until(.73)))
        if ag_result.get("status")=="ok": candidates["automl"]=ag_result
        else: summary["methods"]["automl"]={"status":ag_result.get("status"),"reason":ag_result.get("error",ag_result.get("reason"))}

        # Все методы замораживаются перед открытием confirm. Ранняя остановка по confirm запрещена.
        confirmed={}
        for method,candidate in candidates.items():
            config=(AutoGluonSpec.from_dict(candidate["config"],task) if candidate["config"]["model"]=="autogluon"
                    else ExperimentSpec.from_dict(candidate["config"],task))
            remaining=len(candidates)-len(confirmed)
            allowance=max(5,budget.until(.91)/max(1,remaining))
            result=run(config,method,phase="confirm",timeout=allowance,source=candidate)
            summary["methods"][method]={"status":result.get("status"),"search_score":candidate["score"],
                "confirmation_score":result.get("score"),"config":candidate["config"],
                "selected_model":result.get("selected_model"),"search_directory":Path(candidate["directory"]).name,
                "confirmation_directory":Path(result["directory"]).name if result.get("directory") else None}
            if result.get("status")=="ok": confirmed[method]=result
            save("confirming")
        if not confirmed:
            raise RuntimeError("Ни одна стратегия не прошла независимую проверку")
        winner_method=min(confirmed,key=lambda method:-confirmed[method]["score"] if task.maximize else confirmed[method]["score"])
        winner=confirmed[winner_method]
        with np.load(Path(winner["directory"])/"predictions.npz",allow_pickle=False) as values:
            write_submission(task,test,values["test"],root/"submission.csv")
        summary.update(winner=winner_method,winner_score=winner["score"],llm_calls=planner.calls,
                       llm_accounted_tokens=planner.tokens,semantic_hypotheses=len(descriptions),
                       optuna_trials=len(study.trials),peak_worker_rss_mb=max([r.get("peak_memory_mb",0) for r in all_results]+[0]))
        config=(AutoGluonSpec.from_dict(winner["config"],task) if winner["config"]["model"]=="autogluon"
                else ExperimentSpec.from_dict(winner["config"],task))
        source=candidates[winner_method] if config.model=="autogluon" else winner
        final=run(config,winner_method,phase="full",timeout=budget.until(.985),source=source)
        if final.get("status")=="ok":
            with np.load(Path(final["directory"])/"predictions.npz",allow_pickle=False) as values:
                write_submission(task,test,values["test"],root/"submission.csv")
        summary["final_refit_completed"]=final.get("status")=="ok"
        summary["experiments"]=len(all_results)
        save("completed")
        journal.event("comparison_completed",winner=winner_method,score=winner["score"])
        print(json.dumps(summary["methods"],ensure_ascii=False,indent=2),flush=True)
        return 0
    except Exception as error:
        summary.update(error_type=type(error).__name__,reason=str(error)[-1600:])
        save("failed")
        traceback.print_exc()
        return 1
    finally:
        timer.cancel()


if __name__=="__main__":
    raise SystemExit(main())
