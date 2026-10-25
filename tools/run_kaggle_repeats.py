"""Последовательные повторы seed с фактическим аудитом Docker, без сети/API."""
from pathlib import Path
import argparse
import json
import subprocess
import threading
import uuid

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("case",type=Path)
    parser.add_argument("--seeds",type=int,nargs="+",default=[44,45])
    args=parser.parse_args()
    case=args.case.resolve(strict=True)
    if not 1<=len(args.seeds)<=4 or len(set(args.seeds))!=len(args.seeds) or any(s<0 or s>=2**31 for s in args.seeds):
        raise ValueError("Недопустимый набор повторов")
    existing={json.loads(p.read_text(encoding="utf-8")).get("seed") for p in case.glob(".kaggle_repeat/*/summary.json")}
    if existing & set(args.seeds):
        raise ValueError("Этот seed уже проверялся: повторный запуск и выбор удачной попытки запрещены")
    for seed in args.seeds:
        name="alfa-kaggle-repeat-"+uuid.uuid4().hex[:10]
        command=["docker","run","--detach","--name",name,"--network=none","--cpus=8","--memory=29g",
                 "--memory-swap=29g","--read-only","--tmpfs","/tmp:rw,size=2g","--mount",
                 f"type=bind,source={case},target=/task","--mount",
                 f"type=bind,source={ROOT/'tools'/'repeat_kaggle.py'},target=/helper/repeat_kaggle.py,readonly",
                 "--env","PYTHONDONTWRITEBYTECODE=1","--entrypoint","python3","alfa-research:local",
                 "/helper/repeat_kaggle.py","/task","--seed",str(seed),"--time-limit","900"]
        timer=threading.Timer(960,lambda:subprocess.run(["docker","stop","-t","3",name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL))
        timer.daemon=True
        try:
            subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
            host=json.loads(subprocess.check_output(["docker","inspect","--format","{{json .HostConfig}}",name]))
            audit={"container":name,"cpu_total":host["NanoCpus"]/1e9,"memory_bytes_total":host["Memory"],
                   "memory_swap_bytes":host["MemorySwap"],"network":host["NetworkMode"],"readonly_root":host["ReadonlyRootfs"],
                   "source":"docker inspect во время активного повтора","seed":seed}
            assert audit["cpu_total"]<=8 and audit["memory_bytes_total"]<=32000000000 and audit["network"]=="none"
            timer.start()
            subprocess.run(["docker","logs","--follow",name],check=True)
            code=int(subprocess.check_output(["docker","wait",name]).strip())
            paths=sorted((case/".kaggle_repeat").glob(f"*_seed{seed}/summary.json"))
            if not paths:
                raise RuntimeError("Повтор не сохранил отчёт")
            (paths[-1].parent/"resource_audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf-8")
            if code or json.loads(paths[-1].read_text(encoding="utf-8"))["status"]!="completed":
                raise RuntimeError("Повтор не завершён: "+str(seed))
        finally:
            timer.cancel()
            subprocess.run(["docker","rm","--force",name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


if __name__=="__main__":
    main()
