"""Worker предложений: тот же watchdog и формат результатов, что у текущего агента."""
from pathlib import Path
import json
import pickle
import sys
import time
import traceback


def main(request_path):
    started=time.monotonic();path=Path(request_path).resolve()
    try:
        import numpy as np
        from threadpoolctl import threadpool_limits
        from contracts import TaskSpec
        from data_io import encode_target
        from experiment_store import atomic_json
        from metrics import evaluate
        from proposal_models import ProposalSpec,train_proposal
        from task_reader import read_csv
        request=json.loads(path.read_text(encoding="utf-8"))
        task=TaskSpec.from_dict(request["task"])
        config=ProposalSpec.from_dict(request["config"],task)
        train=read_csv(task.train_path,task.id_columns+task.categorical_columns)
        test=read_csv(task.test_path,task.id_columns+task.categorical_columns)
        y=encode_target(task,train)
        with np.load(request["split_path"],allow_pickle=False) as values:
            tr,va=values["train"],values["valid"]
        if not len(tr) or np.intersect1d(tr,va).size or any(i<0 or i>=len(train) for i in np.r_[tr,va]):
            raise ValueError("Некорректное разбиение")
        with threadpool_limits(limits=1,user_api="blas"):
            bundle,valid,predicted,details=train_proposal(task,config,request["profile"],train,test,y,tr,va,
                request["threads"],request.get("refit",False),request.get("fixed_iterations"),
                max(.1,request.get("fit_time_limit_seconds",600)-(time.monotonic()-started)))
        if len(predicted)!=len(test) or not np.isfinite(predicted).all() or not np.isfinite(valid).all():
            raise ValueError("Некорректные предсказания")
        metrics=evaluate(task,y[va],valid) if len(va) else {}
        np.savez_compressed(path.parent/"predictions.npz",valid=valid,test=predicted,valid_indices=va)
        with (path.parent/"model.pkl").open("wb") as stream:
            pickle.dump(bundle,stream,protocol=4)
        atomic_json(path.parent/"result.json",{"status":"ok","score":metrics.get(task.metric),"metrics":metrics,
            **details,"elapsed_seconds":time.monotonic()-started})
        return 0
    except Exception as error:
        from experiment_store import atomic_json
        atomic_json(path.parent/"result.json",{"status":"failed","error_type":type(error).__name__,"error":str(error)[-1800:],"elapsed_seconds":time.monotonic()-started})
        traceback.print_exc()
        return 1


if __name__=="__main__":
    raise SystemExit(main(sys.argv[1]))
