"""Linux-сравнение с закрытой сетью и единственным разрешённым API через CONNECT.

python tools/run_comparison.py runs/public/regression/task.md --key-file /path/APIkey.txt
"""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from package_solution import MODULES


def package(path):
    members=[name for name in MODULES if name!='solution.py']+['comparison.py','strategy_specs.py','strategy_worker.py']
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in members:
            info=zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644<<16
            archive.writestr(info,(ROOT/name).read_bytes())
        wrapper=('from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parent))\n'
                 'from comparison import main\nraise SystemExit(main())\n')
        info=zipfile.ZipInfo('solution.py',date_time=(2026,1,1,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED
        info.external_attr=0o644<<16
        archive.writestr(info,wrapper)
    if path.stat().st_size>100*1024**2:
        raise ValueError('Архив превышает 100 МБ')
    return path


def docker(*args,**kwargs):
    return subprocess.run(['docker',*args],check=True,**kwargs)


def audit_runtime(container,proxy,network,destination):
    configs=[]
    for name in (container,proxy):
        configs.append(json.loads(subprocess.check_output(
            ['docker','inspect','--format','{{json .HostConfig}}',name])))
    net=json.loads(subprocess.check_output(['docker','network','inspect',network]))[0]
    audit={'cpu_total':sum(c['NanoCpus'] for c in configs)/1e9,
           'memory_bytes_total':sum(c['Memory'] for c in configs),'internal_network':net['Internal']}
    if not(audit['cpu_total']<=8 and audit['memory_bytes_total']<=32000000000 and audit['internal_network']):
        raise RuntimeError('Лимиты стенда не соответствуют ТЗ')
    (destination/'resource_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    code="""
from pathlib import Path
import importlib.metadata as m,json,re,hashlib
path=Path('/task/requirements.txt'); installed={}; unused={}; mismatches=[]
for line in path.read_text(encoding='utf-8').splitlines():
 line=line.strip()
 if not line or line.startswith('#'): continue
 name,expected=re.sub(r'\\[[^\\]]*\\]','',line).split('==')
 try: actual=m.version(name)
 except m.PackageNotFoundError: unused[name]=expected;continue
 installed[name]=actual
 if actual!=expected: mismatches.append({'name':name,'expected':expected,'actual':actual})
result={'installed_approved':installed,'unused_not_installed':unused,'mismatches':mismatches,'requirements_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
Path('/task/dependency_audit.json').write_text(json.dumps(result,indent=2))
assert not mismatches,mismatches
"""
    docker('exec',container,'python3','-c',code,stdout=subprocess.DEVNULL)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('task')
    parser.add_argument('--key-file')
    parser.add_argument('--image',default='alfa-research:local')
    parser.add_argument('--task-limit',type=int,default=10800)
    parser.add_argument('--control-trials',type=int,default=8)
    parser.add_argument('--optuna-trials',type=int,default=12)
    parser.add_argument('--feature-rounds',type=int,default=2)
    parser.add_argument('--iterations',type=int,default=2400)
    parser.add_argument('--trial-seconds',type=int,default=200)
    parser.add_argument('--automl-seconds',type=int,default=1800)
    parser.add_argument('--branch-seconds',type=int,default=1800)
    parser.add_argument('--no-llm',action='store_true')
    args=parser.parse_args()
    if not 30<=args.task_limit<=10800:
        raise ValueError('Лимит задачи превышает ТЗ')
    source=Path(args.task).resolve()
    tag=uuid.uuid4().hex[:10]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    destination=ROOT/'runs'/'comparison'/f'{source.parent.name}_{stamp}_{tag}'
    destination.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source,destination/'task.md')
    for name in ('train.csv','test.csv','data_dictionary.csv','sample_submission.csv'):
        if (source.parent/name).exists():
            shutil.copyfile(source.parent/name,destination/name)
    archive=package(ROOT/'dist'/'alfa_comparison.zip')
    with zipfile.ZipFile(archive) as z:
        if any('/' in name or '\\' in name for name in z.namelist()):
            raise ValueError('Вложенные пути в ZIP')
        z.extractall(destination)
    environment=os.environ.copy()
    if args.key_file:
        environment['OPENAI_API_KEY']=Path(args.key_file).read_text(encoding='utf-8-sig').strip()
    if not args.no_llm and not environment.get('OPENAI_API_KEY'):
        raise ValueError('Для семантических гипотез требуется ключ организаторов')
    network,proxy,study='alfa-internal-'+tag,'alfa-gateway-'+tag,'alfa-study-'+tag
    manifest={'network':network,'proxy':proxy,'container':study,'task_directory':str(destination),
              'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
              'cpu_total':8,'memory_total_limit':'29 GiB + 128 MiB (<32 GB)',
              'allowed_host':'api.alfahack-gateway26.ru','task_limit_seconds':args.task_limit}
    (destination/'docker_objects.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    try:
        docker('network','create','--internal',network,stdout=subprocess.DEVNULL)
        docker('run','--detach','--rm','--name',proxy,'--network',network,'--network-alias','llm-gateway',
               '--cpus=.5','--memory=128m','--read-only','--mount',
               f'type=bind,source={ROOT/"tools"/"gateway_proxy.py"},target=/proxy.py,readonly',
               '--entrypoint','python3',args.image,'/proxy.py',stdout=subprocess.DEVNULL)
        docker('network','connect','bridge',proxy)
        command=['run','--rm','--name',study,'--network',network,'--cpus=7.5','--memory=29g',
                 '--read-only','--tmpfs','/tmp:rw,size=2g','--mount',f'type=bind,source={destination},target=/task',
                 '--env','OPENAI_API_KEY','--env','OPENAI_MODEL=qwen/qwen3.8-27b',
                 '--env','OPENAI_BASE_URL=https://api.alfahack-gateway26.ru/v1',
                 '--env','HTTP_PROXY=http://llm-gateway:8080','--env','HTTPS_PROXY=http://llm-gateway:8080',
                 '--env','NO_PROXY=','--env','ALFA_RESTRICTED_NETWORK=1','--env','PYTHONDONTWRITEBYTECODE=1',
                 '--entrypoint','python3',args.image,'/task/solution.py','task.md',
                 '--time-limit',str(args.task_limit),'--control-trials',str(args.control_trials),
                 '--optuna-trials',str(args.optuna_trials),'--feature-rounds',str(args.feature_rounds),
                 '--max-iterations',str(args.iterations),'--trial-seconds',str(args.trial_seconds),
                 '--branch-seconds',str(args.branch_seconds),'--automl-seconds',str(args.automl_seconds)]
        if args.no_llm: command.append('--no-llm')
        print('Сравнение: '+str(destination),flush=True)
        with (destination/'console.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(['docker',*command],env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                     text=True,encoding='utf-8',errors='replace')
            audited=False
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
                if not audited:
                    audit_runtime(study,proxy,network,destination)
                    audited=True
            code=process.wait(timeout=args.task_limit+60)
        if code:
            raise RuntimeError(f'Сравнение завершилось с кодом {code}')
    finally:
        subprocess.run(['docker','logs',proxy],stdout=(destination/'gateway.log').open('w',encoding='utf-8'),stderr=subprocess.STDOUT)
        for name in (study,proxy):
            subprocess.run(['docker','rm','--force',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        subprocess.run(['docker','network','rm',network],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    summaries=sorted(destination.glob('.comparison/*/comparison_summary.json'))
    if not summaries:
        raise RuntimeError('Отсутствует результат сравнения')
    summary=json.loads(summaries[-1].read_text(encoding='utf-8'))
    if any(summary['methods'].get(method,{}).get('status')!='ok' for method in ('control','optuna','semantic','automl')):
        raise RuntimeError('Не все стратегии завершили подтверждающую проверку')
    if not args.no_llm and not summary.get('semantic_hypotheses'):
        raise RuntimeError('Не проверена ни одна допустимая семантическая гипотеза')
    print(json.dumps({'status':summary['status'],'winner':summary.get('winner'),
                      'methods':summary['methods'],'task_directory':str(destination)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
