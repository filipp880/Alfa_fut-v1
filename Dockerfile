FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=7 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /agent
COPY requirements-dev.txt /agent/requirements-dev.txt
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY *.py /agent/
WORKDIR /task
ENTRYPOINT ["python3", "/agent/solution.py"]
CMD ["task.md"]
