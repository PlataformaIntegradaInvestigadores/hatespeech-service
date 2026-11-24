# Imagen ligera de Python; las librerías CUDA llegan con las wheels de PyTorch,
# así que no hace falta una imagen base nvidia/cuda (el host aporta el driver).
FROM python:3.11-slim

# Por defecto, imagen sin CUDA: solo backend mock (CI, staging, dev sin GPU).
# docker-compose.gpu.yml pasa INSTALL_GPU=true para el modelo real.
ARG INSTALL_GPU=false
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
ARG TORCH_VERSION=2.14.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
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

# torch >= 2.14 despacha algunas ops (p. ej. RoPE) a kernels Triton que se compilan con un
# compilador C que la imagen slim no trae. Se fuerza la ruta eager, la misma con la que se
# evaluó el adapter (entorno Windows sin Triton).
ENV TORCH_DISABLE_NATIVE_JIT=1

USER appuser

EXPOSE 8008

# Mismo puerto dentro y fuera del contenedor, como el resto de servicios (8008).
# start-period largo: cargar Qwen2.5-3B desde un volumen puede tardar minutos.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8008/health', timeout=5)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8008"]
