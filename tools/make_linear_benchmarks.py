"""Синтетические задачи; истинные тестовые ответы остаются только у проверяющего."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd

def main():
    for classification in (False,True):
        rng=np.random.default_rng(271026+int(classification))
        n,m,p=1500,400,12
        x=rng.normal(size=(n+m,p))
        if not classification:x[n:]*=2
        if classification:
            w=rng.normal(size=(p,3))
            labels=np.array(['a','b','c'])
            y=labels[(x@w+.3*rng.normal(size=(n+m,3))).argmax(axis=1)]
        else:y=20+x@np.arange(1,p+1)+rng.normal(scale=.2,size=n+m)
        name='linear_classification' if classification else 'linear_regression'
        out=ROOT/'runs'/'linear_benchmarks'/name;out.mkdir(parents=True,exist_ok=True)
        data=pd.DataFrame(x,columns=[f'x{i}' for i in range(p)])
        data.insert(0,'id',[f'row_{i:05}' for i in range(n+m)])
        train=data.iloc[:n].copy();train['target']=y[:n]
        train.to_csv(out/'train.csv',index=False)
        data.iloc[n:].to_csv(out/'test.csv',index=False)
        pd.DataFrame({'id':data.iloc[n:]['id'],'target':y[:m]}).to_csv(out/'sample_submission.csv',index=False)
        pd.DataFrame({'feature':list(data.columns)+['target'],
            'role':['id']+['feature']*p+['target']}).to_csv(out/'data_dictionary.csv',index=False)
        metric='Macro F1' if classification else 'RMSE'
        kind='multiclass classification' if classification else 'regression'
        (out/'task.md').write_text(f'Task: {kind}. Target: target. Metric: {metric}.\n'
            'Train: train.csv; test: test.csv; dictionary: data_dictionary.csv.\n'
            'The id column is an identifier. Output submission.csv with id and target.\n',encoding='utf-8')
        np.savez_compressed(out/'oracle_test.npz',target=y[n:])
        (out/'generator.json').write_text(json.dumps({'seed':271026+int(classification),
            'train_rows':n,'test_rows':m,'features':p,'oracle_not_mounted':True}),encoding='utf-8')
        print(out)
if __name__=='__main__':main()
