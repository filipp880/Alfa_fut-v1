"""Запуск новых предложений из двух файлов в полностью офлайн Linux-стенде."""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import uuid
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from package_solution import MODULES
from run_kaggle import audit_dependencies

EXTRA=("historical_features.py","kaggle_features.py","kaggle_specs.py","kaggle_worker.py","kaggle_xgb.py",
       "proposal_models.py","proposal_worker.py","proposal_catalog.py","proposal_research.py")


def package(path):
    members=list(dict.fromkeys([n for n in MODULES if n!="solution.py"]+list(EXTRA)))
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        content={n:(ROOT/n).read_bytes() for n in members}
        content["solution.py"]=b"from proposal_research import main\nraise SystemExit(main())\n"
        for name,data in content.items():
            info=zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            archive.writestr(info,data)
    assert path.stat().st_size<100*1024**2
    return path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--time-limit",type=int,default=10800)
    parser.add_argument("--trial-seconds",type=int,default=600)
    args=parser.parse_args()
    if not 60<=args.time_limit<=10800:raise ValueError("Превышен лимит ТЗ")
    source=Path(args.task).resolve();tag=uuid.uuid4().hex[:10]
    destination=ROOT/"runs"/"ai_proposals"/(source.parent.name+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+tag)
    destination.mkdir(parents=True)
    for name in ("task.md","train.csv","test.csv","data_dictionary.csv","sample_submission.csv"):
        shutil.copyfile(source.parent/name,destination/name)
    archive=package(ROOT/"dist"/"alfa_ai_proposals.zip")
    shutil.copyfile(archive,destination/"source.zip")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() or any("/" in n or "\\" in n for n in bundle.namelist()):raise ValueError("Неверный ZIP")
        bundle.extractall(destination)
    materials={}
    for name in ("GPT_ANS.txt","Qwen_ANS.txt"):
        path=Path("C:/Users/PC/Desktop")/name
        if path.exists():
            content=path.read_bytes();materials[name]={"sha256":hashlib.sha256(content).hexdigest(),"bytes":len(content),"lines":len(content.decode("utf-8-sig").splitlines())}
    (destination/"proposal_sources.json").write_text(json.dumps(materials,indent=2),encoding="utf-8")
    container="alfa-proposals-"+tag
    manifest={"container":container,"directory":str(destination),"network":"none","cpu_total":8,
        "memory_bytes_max":29*1024**3,"time_limit_seconds":args.time_limit,"image":"alfa-research:local",
        "archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),"sources":materials}
    (destination/"docker_objects.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    command=["docker","run","--name",container,"--network=none","--cpus=8","--memory=29g","--memory-swap=29g","--read-only",
        "--tmpfs","/tmp:rw,size=2g","--mount",f"type=bind,source={destination},target=/task","--env","PYTHONDONTWRITEBYTECODE=1",
        "--entrypoint","python3","alfa-research:local","/task/solution.py","task.md","--time-limit",str(args.time_limit),"--trial-seconds",str(args.trial_seconds)]
    print("Новые предложения: "+str(destination),flush=True)
    timer=threading.Timer(args.time_limit+60,lambda:subprocess.run(["docker","stop","-t","3",container],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL))
    timer.daemon=True;timer.start()
    try:
        with (destination/"console.log").open("w",encoding="utf-8") as log:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace")
            audited=False
            for line in process.stdout:
                print(line,end="",flush=True);log.write(line);log.flush()
                if not audited:
                    host=json.loads(subprocess.check_output(["docker","inspect","--format","{{json .HostConfig}}",container]))
                    audit={"cpu_total":host["NanoCpus"]/1e9,"memory_bytes_total":host["Memory"],"memory_swap_bytes":host["MemorySwap"],"network":host["NetworkMode"],"readonly_root":host["ReadonlyRootfs"]}
                    assert audit["cpu_total"]<=8 and audit["memory_bytes_total"]<=32000000000 and audit["network"]=="none"
                    (destination/"resource_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
                    audit_dependencies(container);audited=True
            code=process.wait(timeout=30)
        if code:raise RuntimeError("Эксперимент завершился с кодом "+str(code))
    finally:
        timer.cancel();subprocess.run(["docker","rm","--force",container],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    paths=sorted(destination.glob(".ai_proposals/*/summary.json"))
    result=json.loads(paths[-1].read_text(encoding="utf-8")) if paths else {}
    if result.get("status")!="completed":raise RuntimeError("Исследование не завершено: "+str(result.get("status")))
    print(json.dumps({"directory":str(destination),"baseline":result["baseline"],"selected":result["selected"],"seconds":result["elapsed_seconds"]},ensure_ascii=False),flush=True)


if __name__=="__main__":main()
