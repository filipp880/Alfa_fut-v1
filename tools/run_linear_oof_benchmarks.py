"""Изолированные A/B/C-прогоны, максимум два контейнера по 4 CPU одновременно."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['classification','synthetic'])
    args=parser.parse_args()
    out=ROOT/'runs'/'linear_oof_benchmarks';out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for task in (['public/classification'] if args.stage=='classification' else
                 ['linear_benchmarks/linear_regression','linear_benchmarks/linear_classification']):
        for arm in (['B','C'] if args.stage=='classification' else ['A','B','C']):
            jobs.append((task,arm))
    def run(job):
        task,arm=job;label=task.split('/')[-1]+'_'+arm
        archive='dist/alfa_solution_v3_audit_candidate.zip' if arm=='A' else 'dist/alfa_solution_v4_candidate.zip'
        command=[sys.executable,str(ROOT/'tools/run_release.py'),str(ROOT/'runs'/task/'task.md'),
            '--zip',str(ROOT/archive),'--cpus','4','--memory-gb','14',
            '--max-experiments','18' if args.stage=='classification' else '6']
        if arm=='B':command.append('--no-oof')
        print('Старт '+label,flush=True)
        log=out/(label+'.log')
        with log.open('w',encoding='utf-8') as stream:
            process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
        if process.returncode:raise RuntimeError('Неудачный прогон '+label+': '+str(log))
        record=json.loads(log.read_text(encoding='utf-8').splitlines()[-1])
        record.update(arm=arm,task_source=task)
        print(json.dumps(record,ensure_ascii=False),flush=True)
        return record
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,jobs))
    (out/('index_'+args.stage+'.json')).write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
