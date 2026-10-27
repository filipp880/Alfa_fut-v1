"""Проверка фактических моделей собственных локальных прогонов; новых fit нет."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import pickle
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import joblib
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def inspect_model(model,kind,count,seed):
    if kind=='catboost':
        actual=int(model.tree_count_);assert actual==count
        assert model.get_all_params()['random_seed']==seed
        return {'family':kind,'trees':actual,'seed':seed}
    if kind=='lightgbm':
        booster=model.booster_ if hasattr(model,'booster_') else model
        actual=int(booster.current_iteration());assert actual==count
        params=model.get_params() if hasattr(model,'get_params') else booster.params
        assert int(params.get('random_state',params.get('seed',0)))==seed
        return {'family':kind,'iterations':actual,'seed':seed}
    if kind=='extra_trees':
        assert len(model.estimators_)==count and model.random_state==seed
        return {'family':kind,'trees':len(model.estimators_),'seed':seed}
    if kind=='linear':
        estimator=model.named_steps['estimator']
        assert np.isfinite(estimator.coef_).all()
        return {'family':kind,'estimator':type(estimator).__name__,
                'coefficient_shape':list(estimator.coef_.shape),
                'optimizer_iterations':np.asarray(getattr(estimator,'n_iter_',[])).tolist()}
    raise ValueError('Неизвестный тип модели')

def audit(case):
    case=Path(case).resolve()
    if not case.is_relative_to(ROOT/'runs/release'):raise ValueError('Только собственные runs/release')
    work=sorted(case.glob('.alfa_runs/*/summary.json'))[-1].parent
    jobs=[]
    for path in sorted(work.rglob('controller_result.json')):
        r=read(path);req=read(path.parent/'request.json')
        if r['status']!='ok' or not req.get('refit'):continue
        c=req['config'];kind=c['model'];count=req['fixed_iterations'];seed=c['seed']
        assert count==r['best_iterations'] and not r.get('evaluation_used_for_fit',False)
        components=[]
        if kind=='extra_trees':
            location=path.parent/'model.joblib';model=joblib.load(location)
            assert model.config.to_dict()==c
            components.append(inspect_model(model.model,kind,c['forest_trees'],seed))
            del model
        else:
            location=path.parent/'model.pkl'
            with location.open('rb') as stream:bundle=pickle.load(stream)
            if kind=='history_router':
                assert bundle['config']==c
                for i,(history,forest) in enumerate(zip(bundle['history_models'],bundle['forest_models'])):
                    components.append(inspect_model(history['model'],'catboost',count,seed+i))
                    components.append(inspect_model(forest.model,'extra_trees',256,seed+i))
                    if c['fallback']=='forest_huber':
                        components.append(inspect_model(bundle['huber_models'][i]['model'],'lightgbm',c['fallback_iterations'],seed+i))
            else:
                assert bundle['config']==c
                components.append(inspect_model(bundle['model'],kind,count,seed))
            del bundle
        jobs.append({'directory':path.parent.relative_to(case).as_posix(),'phase':r['phase'],
            'config_key':r['config_key'],'model_sha256':sha(location),'components':components})
        gc.collect()
    if not jobs:raise ValueError('Нет фиксированных моделей')
    return {'directory':case.relative_to(ROOT).as_posix(),'status':'verified','jobs':jobs,'new_fits':0,'new_api_calls':0}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',default='docs/LINEAR_OOF_RESULTS.json')
    p.add_argument('--output',default='docs/LINEAR_OOF_MODEL_AUDIT.json');args=p.parse_args()
    results=[]
    for r in read(ROOT/args.index):
        if r['arm']=='A':continue
        print('Модели '+r['directory'],flush=True);results.append(audit(ROOT/r['directory']))
    (ROOT/args.output).write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Все фактические модели проверены',flush=True)
