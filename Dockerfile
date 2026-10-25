FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=7 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /agent
COPY requirements-dev.txt requirements.txt /agent/
RUN python -c "from pathlib import Path; import re; p=Path('/agent'); lines=[re.sub(r'\[[^\]]*\]','',s.strip()) for s in (p/'requirements.txt').read_text(encoding='utf-8').splitlines() if s.strip() and not s.lstrip().startswith('#')]; Path('/tmp/approved-constraints.txt').write_text('\n'.join(lines))"
RUN pip install --no-cache-dir -c /tmp/approved-constraints.txt -r requirements-dev.txt
COPY *.py /agent/
WORKDIR /task
ENTRYPOINT ["python3", "/agent/solution.py"]
CMD ["task.md"]
