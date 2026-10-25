"""AutoGluon: разрешённый список моделей и независимая проверка после refit_full."""
from pathlib import Path
import json
import sys
import time
import traceback


def main(request_path: str) -> int:
    path = Path(request_path).resolve()
    request = json.loads(path.read_text(encoding="utf-8"))
    if request.get("config",{}).get("model") != "autogluon":
        from worker import main as standard_main
        return standard_main(request_path)
    started = time.monotonic()
    try:
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import StratifiedGroupKFold,GroupKFold
        from autogluon.tabular import TabularPredictor
        from contracts import TaskSpec
        from strategy_specs import AutoGluonSpec
        from task_reader import read_csv
        from data_io import encode_target,feature_groups
        from models import sample_weights
        from metrics import evaluate,normalized_probabilities
        from experiment_store import atomic_json
        task = TaskSpec.from_dict(request["task"])
        config = AutoGluonSpec.from_dict(request["config"],task)
        options = request.get("job_options",{})
        train = read_csv(task.train_path,task.id_columns+task.categorical_columns)
        test = read_csv(task.test_path,task.id_columns+task.categorical_columns)
        y = encode_target(task,train)
        with np.load(request["split_path"],allow_pickle=False) as split:
            tr,va = split["train"],split["valid"]
        output_valid_indices = va.copy()
        requested_train_rows = len(tr)
        rebuild_weighted_full = False
        if set(tr.tolist()) & set(va.tolist()):
            raise ValueError("Пересечение train и validation")
        metric = {"rmse":"root_mean_squared_error","mae":"mean_absolute_error","mse":"mean_squared_error",
                  "r2":"r2","macro_f1":"f1_macro","weighted_f1":"f1_weighted","f1":"f1",
                  "accuracy":"accuracy","balanced_accuracy":"balanced_accuracy","log_loss":"log_loss",
                  "roc_auc":"roc_auc" if len(task.classes)==2 else "roc_auc_ovr_macro"}.get(task.metric)
        if metric is None:
            raise ValueError("Метрика не поддерживается исследуемой веткой AutoGluon")
        source = options.get("source_directory")
        if source:
            source_path = Path(source).resolve()
            allowed_root = path.parent.parent.resolve()
            if not source_path.is_relative_to(allowed_root):
                raise ValueError("Источник AutoGluon находится вне текущего запуска")
            source_request=json.loads((source_path/"request.json").read_text(encoding="utf-8"))
            if source_request["task"]["train_path"]!=task.train_path:
                raise ValueError("Попытка переноса модели между задачами")
            with np.load(source_request["split_path"],allow_pickle=False) as data:
                original_indices=np.union1d(data["train"],data["valid"])
            if not options.get("use_all_training") and set(original_indices)!=set(tr):
                raise ValueError("Данные refit не совпадают с train-частью подтверждения")
            recipe=json.loads((source_path/"ag_recipe.json").read_text(encoding="utf-8"))
            if (options.get("use_all_training") and recipe["weight_name"]
                    and len(np.setdiff1d(tr,original_indices))):
                # 1.3.1 rejects sample_weight + train_data_extra. Fresh public fit,
                # then refit its cached train+tuning; no unsupported cache mutation.
                with np.load(source_request["split_path"],allow_pickle=False) as data:
                    va=data["valid"].copy()
                if not set(va).issubset(set(tr)):
                    raise ValueError("Исходная tuning-часть отсутствует в полном обучении")
                tr=np.setdiff1d(tr,va)
                rebuild_weighted_full=True
                source=None
        if source:
            predictor = TabularPredictor.load(str(source_path/"ag_models"))
            predictor = predictor.clone(path=str(path.parent/"ag_models"),return_clone=True)
            original = predictor.model_best
            extra=None
            if options.get("use_all_training"):
                extra_indices=np.setdiff1d(tr,original_indices)
                if len(extra_indices):
                    recipe=json.loads((source_path/"ag_recipe.json").read_text(encoding="utf-8"))
                    extra=train.iloc[extra_indices][task.feature_columns].copy()
                    extra[recipe["label"]]=y[extra_indices]
                    if recipe["weight_name"]:
                        extra[recipe["weight_name"]]=[recipe["class_weights"].get(str(int(v)),1.) for v in y[extra_indices]]
            mapping = predictor.refit_full(model=original,set_best_to_refit_full=False,
                        train_data_extra=extra,num_cpus=request["threads"],num_gpus=0,fit_strategy="sequential")
            model_name = mapping[original]
            # refit_full использует train+tuning. Их индексы проверены контроллером.
        else:
            x_train = train.iloc[tr][task.feature_columns].copy()
            x_valid = train.iloc[va][task.feature_columns].copy()
            label = "__alfa_target"
            while label in x_train:
                label += "_"
            x_train[label],x_valid[label] = y[tr],y[va]
            weight_name = None
            class_weights={}
            weights = sample_weights(y[tr],len(task.classes),config.class_weight_power) if task.classification else None
            if weights is not None:
                weight_name = "__alfa_weight"
                while weight_name in x_train:
                    weight_name += "_"
                x_train[weight_name] = weights
                class_weights={str(int(c)):float(weights[np.flatnonzero(y[tr]==c)[0]]) for c in np.unique(y[tr])}
                x_valid[weight_name] = [class_weights[str(int(c))] for c in y[va]]
            atomic_json(path.parent/"ag_recipe.json",{"label":label,"weight_name":weight_name,"class_weights":class_weights})
            group_name,folds = None,0
            if config.bag_folds and not request["profile"].get("time_column"):
                groups = feature_groups(task,train)[tr]
                minimum = min(len(np.unique(groups[y[tr]==c])) for c in np.unique(y[tr])) if task.classification else len(np.unique(groups))
                folds = min(config.bag_folds,minimum)
                if folds>=2:
                    cv = (StratifiedGroupKFold(folds,shuffle=True,random_state=config.seed) if task.classification
                          else GroupKFold(folds))
                    fold_ids = np.zeros(len(tr),dtype=int)
                    for fold,(_,indices) in enumerate(cv.split(np.arange(len(tr)),y[tr],groups)):
                        fold_ids[indices] = fold
                    group_name = "__alfa_bag_group"
                    while group_name in x_train:
                        group_name += "_"
                    x_train[group_name] = fold_ids
                else:
                    folds=0
            problem = "binary" if task.classification and len(task.classes)==2 else "multiclass" if task.classification else "regression"
            predictor = TabularPredictor(label=label,problem_type=problem,eval_metric=metric,
                        path=str(path.parent/"ag_models"),sample_weight=weight_name,weight_evaluation=False,
                        groups=group_name,positive_class=task.positive_index if problem=="binary" else None,
                        verbosity=2,learner_kwargs={"label_count_threshold":1})
            # Только разрешённые алгоритмы, все обучаются с нуля. Нет foundation/NN-моделей.
            hyperparameters = {
                "GBM":[{"num_boost_round":config.iterations,"num_leaves":31,"learning_rate":config.learning_rate,
                        "lambda_l2":config.regularization,"min_data_in_leaf":config.min_child_samples,
                        "ag_args":{"name_suffix":"Small"}},
                       {"num_boost_round":config.iterations,"num_leaves":63,"learning_rate":.04,
                        "lambda_l2":8,"min_data_in_leaf":60,"ag_args":{"name_suffix":"Regularized"}}],
                "CAT":[{"iterations":config.iterations,"depth":6,"learning_rate":config.learning_rate,
                        "l2_leaf_reg":config.regularization}],
                "XGB":[{"n_estimators":config.iterations,"max_depth":6,"learning_rate":.05,
                        "reg_lambda":5,"subsample":.85,"colsample_bytree":.9,"tree_method":"hist","device":"cpu"}],
            }
            limit = max(5,request["fit_time_limit_seconds"]-(time.monotonic()-started)-30)
            predictor.fit(x_train,tuning_data=x_valid,hyperparameters=hyperparameters,time_limit=limit,
                          num_cpus=request["threads"],num_gpus=0,memory_limit=24,
                          fit_strategy="sequential",num_bag_folds=folds,num_bag_sets=1,num_stack_levels=0,
                          use_bag_holdout=bool(folds),dynamic_stacking=False,auto_stack=False,
                          calibrate_decision_threshold=False,refit_full=False,
                          ag_args_ensemble={"fold_fitting_strategy":"sequential_local"})
            model_name = predictor.model_best
            if rebuild_weighted_full:
                mapping=predictor.refit_full(model=model_name,set_best_to_refit_full=False,
                    num_cpus=request["threads"],num_gpus=0,fit_strategy="sequential")
                model_name=mapping[model_name]

        def predict(frame):
            if not len(frame):
                return np.empty((0,len(task.classes))) if task.classification else np.empty(0)
            x = frame[task.feature_columns]
            if not task.classification:
                return predictor.predict(x,model=model_name).to_numpy(dtype=float)
            raw = predictor.predict_proba(x,model=model_name,as_multiclass=True)
            result = np.zeros((len(x),len(task.classes)))
            for col in raw.columns:
                result[:,int(col)] = raw[col].to_numpy(dtype=float)
            return normalized_probabilities(result,len(task.classes))

        valid_pred,test_pred = predict(train.iloc[output_valid_indices]),predict(test)
        values = evaluate(task,y[output_valid_indices],valid_pred) if len(output_valid_indices) else {}
        np.savez_compressed(path.parent/"predictions.npz",valid=valid_pred,test=test_pred,
                            valid_indices=output_valid_indices)
        leaderboard = predictor.leaderboard(silent=True).fillna("").to_dict("records")
        atomic_json(path.parent/"result.json",{"status":"ok","score":values.get(task.metric),"metrics":values,
            "n_train":requested_train_rows,"training_seconds":time.monotonic()-started,"prediction_seconds":0,
            "best_iterations":config.iterations,"selected_model":model_name,"leaderboard":leaderboard,
            "elapsed_seconds":time.monotonic()-started,"model_family":"autogluon"})
        return 0
    except Exception as error:
        from experiment_store import atomic_json
        atomic_json(path.parent/"result.json",{"status":"failed","error_type":type(error).__name__,
            "error":str(error)[-1800:],"elapsed_seconds":time.monotonic()-started})
        traceback.print_exc()
        return 1


if __name__=="__main__":
    raise SystemExit(main(sys.argv[1]))
