"""Проверка именно распакованного ZIP в Linux, без доступа в интернет.

python tools/check_linux.py runs/public/classification/task.md --time-limit 900
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import shutil
import subprocess
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from package_solution import build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task')
    parser.add_argument('--image', default='alfa-agent:local')
    parser.add_argument('--time-limit', type=int, default=900)
    parser.add_argument('--max-experiments', type=int, default=8)
    parser.add_argument('--max-iterations', type=int, default=1600)
    args = parser.parse_args()
    source = Path(args.task).resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    destination = ROOT/'runs'/'linux'/f'{source.parent.name}_{stamp}'
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, destination/'task.md')
    for name in ('train.csv','test.csv','data_dictionary.csv','sample_submission.csv'):
        candidate = source.parent/name
        if candidate.exists():
            shutil.copyfile(candidate,destination/name)
    archive = build(ROOT,ROOT/'dist'/'alfa_solution.zip')
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    name = 'alfa-check-'+uuid.uuid4().hex[:10]
    command = ['docker','run','--rm','--name',name,'--cpus=8','--memory=32g','--network=none',
               '--mount',f'type=bind,source={destination},target=/task','--entrypoint','python3',
               args.image,'/task/solution.py','task.md','--time-limit',str(args.time_limit),
               '--max-experiments',str(args.max_experiments),'--max-iterations',str(args.max_iterations),'--no-llm']
    print(f'Проверка ZIP в Linux: {destination}',flush=True)
    try:
        subprocess.run(command,check=True,timeout=args.time_limit+60)
    finally:
        subprocess.run(['docker','rm','--force',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    summaries = sorted(destination.glob('.alfa_runs/*/summary.json'))
    if not summaries:
        raise RuntimeError('Нет отчёта агента')
    summary = json.loads(summaries[-1].read_text(encoding='utf-8'))
    if summary['status']!='completed' or summary.get('fallback_only',True):
        raise RuntimeError(f'Агент не завершил модельный прогон: {summary["status"]}')
    events = [json.loads(line) for line in summaries[-1].with_name('experiments.jsonl').read_text(encoding='utf-8').splitlines()]
    succeeded = {e.get('config',{}).get('model') for e in events if e['event']=='experiment_result' and e.get('status')=='ok'}
    if args.max_experiments>=2 and not {'catboost','lightgbm'} <= succeeded:
        raise RuntimeError('Не подтверждена работа обоих бустингов в контейнере')
    report = {k:summary[k] for k in ('status','elapsed_seconds','search_best_score','refit_completed','versions','python')}
    report.update(selected=summary['selected'],rows=summary['profile']['test_rows'],task_directory=str(destination))
    (destination/'linux_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
