# Immagine del servizio di inferenza. Tre fasi: base (dipendenze), test (con pytest), runtime (predefinita, ultima).
#   docker build -f api/Dockerfile --target test -t irp-api-test .      # esegue i test con `docker run`
#   docker build -f api/Dockerfile --build-arg WITH_TORCH=0 -t irp-api .  # senza PyTorch: solo stadio 1 del triage
FROM python:3.11-slim AS base
ARG SKLEARN_VERSION=
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HOME=/tmp
COPY api/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
 # i modelli sono pickle di scikit-learn: la versione deve coincidere con quella dell'addestramento (vedi i metadati)
 && if [ -n "$SKLEARN_VERSION" ]; then pip install "scikit-learn==$SKLEARN_VERSION"; fi

FROM base AS test
COPY api/requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install -r /tmp/requirements-dev.txt
COPY pytest.ini /app/pytest.ini
COPY src /app/src
COPY tests /app/tests
CMD ["python", "-m", "pytest", "-q"]

FROM base AS runtime
ARG WITH_TORCH=1
# PyTorch solo CPU + transformers per lo stadio 2; con WITH_TORCH=0 il servizio parte con il solo stadio 1
RUN if [ "$WITH_TORCH" = "1" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install "transformers>=4.40,<5"; \
    fi
COPY src/serving /app/serving
ENV PYTHONPATH=/app HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"
CMD ["uvicorn", "serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
