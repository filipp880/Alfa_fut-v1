"""Воспроизводимый Linux-стенд новых гипотез с разрешённым CONNECT-прокси."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from package_solution import MODULES
from run_comparison import audit_runtime

EXTRA=("historical_features.py","kaggle_features.py","kaggle_xgb.py","kaggle_specs.py",
       "kaggle_worker.py","aide_search.py","kaggle_research.py")


def audit_dependencies(container):
    code="""
from pathlib import Path
import importlib.metadata as m,json,re,hashlib
p=Path('/task/requirements.txt'); installed={}; unused={}; bad=[]
for line in p.read_text(encoding='utf-8').splitlines():
 line=line.strip()
 if not line or line.startswith('#'): continue
 name,expected=re.sub(r'\\[[^\\]]*\\]','',line).split('==')
 try: actual=m.version(name)
 except m.PackageNotFoundError: unused[name]=expected;continue
 installed[name]=actual
 if actual!=expected: bad.append({'name':name,'expected':expected,'actual':actual})
result={'installed_approved':installed,'unused_not_installed':unused,'mismatches':bad,'requirements_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
Path('/task/dependency_audit.json').write_text(json.dumps(result,indent=2))
assert not bad,bad
"""
    subprocess.run(["docker","exec",container,"python3","-c",code],check=True,stdout=subprocess.DEVNULL)


def package(path):
    members=list(dict.fromkeys([n for n in MODULES if n!="solution.py"]+list(EXTRA)))
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        content={n:(ROOT/n).read_bytes() for n in members}
        content["solution.py"]=b"from kaggle_research import main\nraise SystemExit(main())\n"
        for name,data in content.items():
            info=zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644<<16
            archive.writestr(info,data)
    if path.stat().st_size>100*1024**2:
        raise ValueError("ZIP превышает ТЗ")
    return path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--key-file")
    parser.add_argument("--image",default="alfa-research:local")
    parser.add_argument("--time-limit",type=int,default=10800)
    parser.add_argument("--trial-seconds",type=int,default=420)
    parser.add_argument("--control-trials",type=int,default=8)
    parser.add_argument("--tree-trials",type=int,default=6)
    parser.add_argument("--no-llm",action="store_true")
    args=parser.parse_args()
    if not 60<=args.time_limit<=10800:
        raise ValueError("Превышен лимит ТЗ")
    source=Path(args.task).resolve()
    tag=uuid.uuid4().hex[:10]
    destination=ROOT/"runs"/"kaggle"/(source.parent.name+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+tag)
    destination.mkdir(parents=True)
    for name in ("task.md","train.csv","test.csv","data_dictionary.csv","sample_submission.csv"):
        shutil.copyfile(source.parent/name,destination/name)
    archive=package(ROOT/"dist"/"alfa_kaggle_research.zip")
    shutil.copyfile(archive,destination/"source.zip")
    with zipfile.ZipFile(archive) as bundle:
        if any("/" in n or "\\" in n for n in bundle.namelist()):
            raise ValueError("Исследовательский ZIP содержит папки")
        bundle.extractall(destination)
    environment=os.environ.copy()
    if args.key_file and not args.no_llm:
        environment["OPENAI_API_KEY"]=Path(args.key_file).read_text(encoding="utf-8-sig").strip()
    if not args.no_llm and not environment.get("OPENAI_API_KEY"):
        raise ValueError("Не задан ключ организаторского шлюза")
    network,proxy,study="alfa-kaggle-net-"+tag,"alfa-kaggle-gateway-"+tag,"alfa-kaggle-study-"+tag
    manifest={"network":network if not args.no_llm else "none","proxy":proxy if not args.no_llm else None,
              "container":study,"task_directory":str(destination),"image":args.image,
              "archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
              "cpu_total":8,"memory_bytes_max":29*1024**3+128*1024**2,
              "time_limit_seconds":args.time_limit,"allowed_host":"api.alfahack-gateway26.ru" if not args.no_llm else None}
    (destination/"docker_objects.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    def docker(*command,**kwargs):
        return subprocess.run(["docker",*command],check=True,**kwargs)
    timer=None
    try:
        if not args.no_llm:
            docker("network","create","--internal",network,stdout=subprocess.DEVNULL)
            docker("run","--detach","--rm","--name",proxy,"--network",network,"--network-alias","llm-gateway",
                   "--cpus=.5","--memory=128m","--memory-swap=128m","--read-only","--mount",
                   f"type=bind,source={ROOT/'tools'/'gateway_proxy.py'},target=/proxy.py,readonly",
                   "--entrypoint","python3",args.image,"/proxy.py",stdout=subprocess.DEVNULL)
            docker("network","connect","bridge",proxy)
        command=["docker","run","--name",study,"--network",network if not args.no_llm else "none",
                 "--cpus=7.5" if not args.no_llm else "--cpus=8","--memory=29g","--memory-swap=29g","--read-only",
                 "--tmpfs","/tmp:rw,size=2g","--mount",f"type=bind,source={destination},target=/task",
                 "--env","PYTHONDONTWRITEBYTECODE=1"]
        if not args.no_llm:
            for value in ("OPENAI_API_KEY","OPENAI_MODEL=qwen/qwen3.8-27b","OPENAI_BASE_URL=https://api.alfahack-gateway26.ru/v1",
                          "HTTP_PROXY=http://llm-gateway:8080","HTTPS_PROXY=http://llm-gateway:8080","NO_PROXY=","ALFA_LLM_TOKEN_LIMIT=12000"):
                command += ["--env",value]
        command += ["--entrypoint","python3",args.image,"/task/solution.py","task.md","--time-limit",str(args.time_limit),
                    "--trial-seconds",str(args.trial_seconds),"--control-trials",str(args.control_trials),"--tree-trials",str(args.tree_trials)]
        if args.no_llm:
            command.append("--no-llm")
        print("Каталог исследования: "+str(destination),flush=True)
        timer=threading.Timer(args.time_limit+60,lambda:subprocess.run(["docker","stop","-t","3",study],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL))
        timer.daemon=True
        timer.start()
        with (destination/"console.log").open("w",encoding="utf-8") as log:
            process=subprocess.Popen(command,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                     text=True,encoding="utf-8",errors="replace")
            audited=False
            for line in process.stdout:
                print(line,end="",flush=True)
                log.write(line);log.flush()
                if not audited:
                    if not args.no_llm:
                        audit_runtime(study,proxy,network,destination)
                    else:
                        host=json.loads(subprocess.check_output(["docker","inspect","--format","{{json .HostConfig}}",study]))
                        audit={"cpu_total":host["NanoCpus"]/1e9,"memory_bytes_total":host["Memory"],"network":host["NetworkMode"]}
                        assert audit["cpu_total"]<=8 and audit["memory_bytes_total"]<=32000000000 and audit["network"]=="none"
                        (destination/"resource_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
                        audit_dependencies(study)
                    audited=True
            code=process.wait(timeout=30)
        if code:
            raise RuntimeError("Исследование завершилось с кодом "+str(code))
    finally:
        if timer:
            timer.cancel()
        if not args.no_llm:
            with (destination/"gateway.log").open("w",encoding="utf-8") as log:
                subprocess.run(["docker","logs",proxy],stdout=log,stderr=subprocess.STDOUT)
        for name in (study,proxy):
            subprocess.run(["docker","rm","--force",name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if not args.no_llm:
            subprocess.run(["docker","network","rm",network],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    paths=sorted(destination.glob(".kaggle_research/*/summary.json"))
    if not paths:
        raise RuntimeError("Отсутствует журнал исследования")
    result=json.loads(paths[-1].read_text(encoding="utf-8"))
    if result["status"]!="completed":
        raise RuntimeError("Исследование не завершено: "+result["status"])
    print(json.dumps({"directory":str(destination),"baseline":result["baseline"],"integrated":result["integrated"],
                      "elapsed_seconds":result["elapsed_seconds"]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
