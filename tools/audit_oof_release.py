"""Независимая реконструкция OOF-матрицы и сетки весов по сохранённым прогнозам."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
from data_io import encode_target,feature_groups,detect_time
from task_reader import load_task
from run_sgr_study import score

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def analyze(case):
    case=Path(case).resolve()
    work=sorted(case.glob('.alfa_runs/*/summary.json'))[-1].parent
    out=work/'oof';summary=read(work/'summary.json')
    if not (out/'status.json').exists():return {'status':'disabled','fit_attempts':0}
    state=read(out/'status.json')
    report={k:state.get(k) for k in ('status','reason','fit_attempts','completed_fits','validated_fits','elapsed_seconds','weights','oof_score')}
    if state['status']!='completed':return report
    task,train,test,_,dictionary=load_task(case/'task.md')
    y=encode_target(task,train);groups=feature_groups(task,train)
    _,times=detect_time(task,train,test,dictionary)
    outer=np.load(work/'split_search.npz')['train']
    configs=read(out/'configs.json')['members']
    assert [c['family'] for c in configs]==['boosting','forest','linear']
    assert sha(out/'configs.json')==state['configs_sha256']
    search={read(p)['config_key']:read(p) for p in work.glob('*/controller_result.json')
        if read(p).get('phase')=='search' and read(p).get('status')=='ok'}
    search_dirs={read(p)['config_key']:p.parent for p in work.glob('*/controller_result.json')
        if read(p).get('phase')=='search' and read(p).get('status')=='ok'}
    for c in configs:
        r=search[c['config_key']]
        assert c['config']==r['config'] and c['best_iterations']==r['best_iterations']
        assert r['score']==c['search_score']
        models={'boosting':{'catboost','lightgbm'},'forest':{'extra_trees'},'linear':{'linear'}}[c['family']]
        best=(max if task.maximize else min)(v['score'] for v in search.values() if v['config']['model'] in models)
        np.testing.assert_allclose(c['search_score'],best,rtol=0,atol=1e-12)
    assert sha(out/'raw_matrix.npz')==state['raw_matrix_sha256']
    with np.load(out/'raw_matrix.npz',allow_pickle=False) as a:
        indices=a['indices'];matrix=a['predictions'];target=a['target'];warmup=a['warmup_indices']
        np.testing.assert_array_equal(a['member_keys'],[c['config_key'] for c in configs])
        np.testing.assert_array_equal(target,y[indices])
    assert np.isin(indices,outer).all() and len(np.unique(indices))==len(indices)
    np.testing.assert_array_equal(np.sort(np.r_[indices,warmup]),np.sort(outer))
    reconstructed=np.full_like(matrix,np.nan);covered=[]
    for fold_dir in sorted(out.glob('fold*')):
        with np.load(fold_dir/'split.npz') as s:tr,va=s['train'],s['valid']
        assert np.isin(np.r_[tr,va],outer).all()
        assert not np.intersect1d(tr,va).size and not np.intersect1d(groups[tr],groups[va]).size
        if times is not None:
            assert times[tr].max()<times[va].min()
            np.testing.assert_array_equal(np.sort(tr),np.sort(outer[times[outer]<times[va].min()]))
        else:np.testing.assert_array_equal(np.sort(tr),np.setdiff1d(outer,va))
        profile=read(fold_dir/'profile.json')
        assert profile['profile_scope']=='fold_train_only' and profile['train_rows']==len(tr)
        if task.classification:
            assert profile['class_counts']==train.iloc[tr][task.target].value_counts().to_dict()
        else:np.testing.assert_allclose(profile['target_summary']['mean'],y[tr].mean(),rtol=1e-12)
        covered.extend(va.tolist())
        jobs=[j for j in state['jobs'] if Path(j['directory']).parts[0]==fold_dir.name]
        assert len(jobs)==3
        for j in jobs:
            member=next(i for i,c in enumerate(configs) if c['config_key']==j['config_key'])
            job=out/j['directory'];request=read(job/'request.json');result=read(job/'controller_result.json')
            requested=Path(request['split_path'].replace('/task',str(case),1))
            assert requested.resolve()==(fold_dir/'split.npz').resolve()
            assert request['profile']==profile
            assert request['config']==configs[member]['config']
            assert request['refit'] and not result.get('evaluation_used_for_fit',False)
            assert request['fixed_iterations']==result['best_iterations']==configs[member]['best_iterations']
            assert result['n_train']==len(tr) and result['status']=='ok'
            assert not result.get('training_stopped_for_budget',False)
            assert sha(job/'predictions.npz')==j['prediction_sha256']
            assert (out/'configs.json').stat().st_mtime_ns<=(job/'request.json').stat().st_mtime_ns
            with np.load(job/'predictions.npz') as a:
                np.testing.assert_array_equal(a['valid_indices'],va)
                prediction=a['valid'];assert len(a['test'])==len(test)
            assert np.isfinite(prediction).all()
            if task.classification:
                assert prediction.shape==(len(va),len(task.classes))
                assert (prediction>=0).all() and (prediction<=1).all()
                np.testing.assert_allclose(prediction.sum(axis=1),1,atol=1e-7)
            np.testing.assert_allclose(score(task,y[va],prediction),result['score'],rtol=1e-10)
            reconstructed[np.searchsorted(indices,va),member]=prediction
    np.testing.assert_array_equal(np.sort(covered),indices)
    np.testing.assert_array_equal(reconstructed,matrix)
    grid={(i/10,j/10,(10-i-j)/10) for i in range(11) for j in range(11-i)}
    grid.add((1/3,1/3,1/3))
    for i in range(3):
        for j in range(i+1,3):
            for w in (.25,.5,.75):
                v=[0.,0.,0.];v[i]=w;v[j]=1-w;grid.add(tuple(v))
    evaluated=[(w,score(task,target,sum(w[i]*matrix[:,i] for i in range(3)))) for w in grid]
    higher=task.metric not in {'rmse','mse','mae','rmsle','mape','log_loss'}
    optimum=(max if higher else min)(v for _,v in evaluated)
    weights=state['weights'];actual=score(task,target,sum(weights[i]*matrix[:,i] for i in range(3)))
    np.testing.assert_allclose(actual,optimum,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(actual,state['oof_score'],rtol=1e-10)
    assert len(grid)==state['grid_points']
    assert (out/'status.json').stat().st_mtime_ns<=(work/'finalists.json').stat().st_mtime_ns
    frozen=read(work/'finalists.json')
    events=[json.loads(line) for line in (work/'experiments.jsonl').read_text(encoding='utf-8').splitlines()]
    evaluations=[e for e in events if e['event']=='oof_plan_evaluated']
    assert len(evaluations)==1
    search_valid=np.load(work/'split_search.npz')['valid']
    def validation_prediction(key):
        with np.load(search_dirs[key]/'predictions.npz') as a:
            np.testing.assert_array_equal(a['valid_indices'],search_valid)
            return a['valid'].copy()
    prediction=sum(w*validation_prediction(c['config_key']) for w,c in zip(weights,configs))
    validation_score=score(task,y[search_valid],prediction)
    np.testing.assert_allclose(validation_score,evaluations[0]['score'],rtol=1e-10,atol=1e-12)
    legacy=frozen['legacy_selected_plan']
    legacy_prediction=sum(w*validation_prediction(k) for w,k in zip(legacy['weights'],legacy['members']))
    legacy_score=score(task,y[search_valid],legacy_prediction)
    np.testing.assert_allclose(legacy_score,legacy['score'],rtol=1e-10,atol=1e-12)
    gain=(validation_score-legacy_score) if task.maximize else (legacy_score-validation_score)
    accepted=gain>.001*max(1,abs(legacy_score))
    assert accepted==frozen['oof_selected']==summary['oof_selected_on_search']
    if accepted:
        assert frozen['selected_plan']['members']==state['selected_plan']['members']
        assert frozen['selected_plan']['weights']==state['selected_plan']['weights']
    report.update(checks_passed=True,rows=len(indices),warmup_rows=len(warmup),grid_points=len(grid),
        effective_families=sum(w>0 for w in weights),selected=summary.get('oof_selected',False),
        search_valid_score=validation_score,legacy_search_valid_score=legacy_score,search_valid_accepted=accepted)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('cases',nargs='+');parser.add_argument('--output',required=True)
    args=parser.parse_args();result={str(Path(p).name):analyze(p) for p in args.cases}
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
