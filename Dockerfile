# Imagen ligera de Python; las librerías CUDA llegan con las wheels de PyTorch,
# así que no hace falta una imagen base nvidia/cuda (el host aporta el driver).
FROM python:3.11-slim

# INSTALL_GPU=false construye una imagen solo con el backend mock (CI / dev sin GPU).
ARG INSTALL_GPU=true
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
ARG TORCH_VERSION=2.14.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY requirements.txt requirements-gpu.txt ./

RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_GPU" = "true" ]; then \
         pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX_URL" \
         && pip install --no-cache-dir -r requirements-gpu.txt; \
       fi

COPY --chown=appuser:appuser . .

RUN mkdir -p /home/appuser/.cache/huggingface && chown -R appuser:appuser /home/appuser/.cache

USER appuser

EXPOSE 8006

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8006"]
