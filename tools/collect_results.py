"""Собрать проверяемый отчёт из завершённых локальных прогонов."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from task_reader import load_task
from data_io import encode_target
from submission import fallback_prediction
from metrics import evaluate


def collect(search_root: Path):
    latest = {}
    for file in search_root.rglob('summary.json'):
        if file.parent.parent.name!='.alfa_runs':
            continue
        summary = json.loads(file.read_text(encoding='utf-8'))
        if summary.get('status')!='completed' or summary.get('fallback_only',True):
            continue
        environment = 'Linux' if 'GCC' in summary.get('python','') else 'Windows'
        key = (summary['task']['problem_type'],summary['task']['target'],environment)
        if key not in latest or file.parent.name>latest[key][0].parent.name:
            latest[key] = (file,summary)
    reports = []
    for key,(file,summary) in sorted(latest.items()):
        task_dir = file.parents[2]
        task,train,test,_,_ = load_task(task_dir/'task.md')
        y = encode_target(task,train)
        events = [json.loads(line) for line in file.with_name('experiments.jsonl').read_text(encoding='utf-8').splitlines()]
        experiments = [e for e in events if e['event']=='experiment_result']
        members = []
        for selected in summary['selected']:
            matches = [e for e in experiments if e.get('status')=='ok' and e.get('phase')==selected['phase']
                       and e.get('config')==selected['config']]
            if not matches:
                raise ValueError('Нет исходного результата выбранной модели')
            members.append(matches[-1])
        prediction = None
        valid_indices = None
        for weight,member in zip(summary['weights'],members):
            folder = PurePosixPath(member['directory'].replace('\\','/')).name
            with np.load(file.parent/folder/'predictions.npz',allow_pickle=False) as values:
                if valid_indices is not None and not np.array_equal(valid_indices,values['valid_indices']):
                    raise ValueError('Несогласованные разбиения ансамбля')
                valid_indices = values['valid_indices'].copy()
                part = weight*values['valid']
                prediction = part if prediction is None else prediction+part
        score = evaluate(task,y[valid_indices],prediction)[task.metric]
        phase = members[0]['phase']
        with np.load(file.parent/f'split_{phase}.npz',allow_pickle=False) as split:
            constant = fallback_prediction(task,y[split['train']],len(split['valid']))
            baseline = evaluate(task,y[split['valid']],constant)[task.metric]
        report = {'environment':key[2],'problem_type':task.problem_type,'target':task.target,
                  'metric':task.metric,'search_score':summary['search_best_score'],
                  'selection_phase':phase,'validation_score':score,'constant_baseline_score':baseline,
                  'elapsed_seconds':summary['elapsed_seconds'],'train_rows':len(train),'test_rows':len(test),
                  'refit_completed':summary['refit_completed'],'weights':summary['weights'],
                  'selected':summary['selected'],'successful_experiments':summary['successful_experiments'],
                  'failed_experiments':len([e for e in experiments if e.get('status')!='ok']),
                  'peak_worker_rss_mb':max([e.get('peak_memory_mb',0) for e in experiments]+[0]),
                  'versions':summary['versions'],'python':summary['python'],
                  'llm_calls':summary.get('llm_calls',0),'llm_accounted_tokens':summary.get('llm_tokens',0),
                  'run_id':file.parent.name}
        reports.append(report)
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs',default='runs')
    args = parser.parse_args()
    reports = collect(ROOT/args.runs)
    destination = ROOT/'docs'
    destination.mkdir(exist_ok=True)
    (destination/'validation_results.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# Результаты локальной проверки','',
      'Метрики вычислены на собственных разбиениях обучающих данных. Это не результаты скрытого теста платформы.',
      'Константный baseline ниже является нашим аварийным ответом, а не бейзлайном организаторов.','',
      '| Среда | Таргет | Метрика | Поиск | Подтверждение | Константа на той же части | Время, с | Пик RSS worker, МБ |',
      '|---|---|---|---:|---:|---:|---:|---:|']
    for report in reports:
        lines.append(f"| {report['environment']} | {report['target']} | {report['metric']} | "
            f"{report['search_score']:.6f} | {report['validation_score']:.6f} | {report['constant_baseline_score']:.6f} | "
            f"{report['elapsed_seconds']:.1f} | {report['peak_worker_rss_mb']:.1f} |")
    lines += ['', 'Подробные конфигурации, версии и идентификаторы прогонов: `validation_results.json`.','',
      'Linux-прогоны выполняются из распакованного ZIP с `--cpus=8 --memory=32g --network=none`.',
      'Windows-прогоны относятся к предыдущей итерации портфеля; таблица не является сравнением производительности операционных систем.',
      'Пик RSS относится к дереву процессов worker и измеряется периодически; родительский процесс учитывается отдельно при назначении лимита.',
      'Сокращённые прогоны ограничивают число экспериментов и деревьев; полный трёхчасовой поиск не заявляется как выполненный.',
      'Для регрессии поиск использует периоды 1–3 → 4, подтверждение — 1–4 → 5.',
      'Для классификации используются отдельные стратифицированные группы повторяющихся признаков.',
      'Выбор модели на подтверждающей части сам по себе не даёт независимой оценки качества на приватных данных.','']
    (destination/'VALIDATION.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('environment','target','validation_score','elapsed_seconds','refit_completed')}
                      for r in reports],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
