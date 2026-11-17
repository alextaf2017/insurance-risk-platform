# Ambiente NLP separato da Spark: PyTorch per CPU + Hugging Face Transformers.
FROM python:3.11-slim

WORKDIR /workspace
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

# PyTorch build solo CPU (molto piu' leggera di quella con CUDA)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Container di sviluppo eseguito come root per semplicita' con i bind mount.
