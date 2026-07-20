# syntax=docker/dockerfile:1
FROM python:3.10-slim

# ---- System dependencies ---------------------------------------------------
# libgl1/libglib2.0-0: required by opencv-python-headless at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python dependencies ----------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Application code ---------------------------------------------------------
COPY . .

# Cache pretrained model weights / torch hub downloads in a writable,
# container-persistent directory. Hugging Face Spaces and Render both give
# the container a writable filesystem at runtime (though it may not persist
# across redeploys) -- mount a volume at this path if you want to avoid
# re-downloading weights on every cold start.
ENV MODEL_WEIGHTS_DIR=/app/model_weights \
    TORCH_HOME=/app/model_weights/torch \
    XDG_CACHE_HOME=/app/model_weights/cache \
    FLASK_ENV=production \
    DEVICE=cpu \
    NO_ALBUMENTATIONS_UPDATE=1 \
    PYTHONUNBUFFERED=1

RUN mkdir -p /app/model_weights /app/static/uploads /app/logs \
    && python -c "from pix2tex.cli import LatexOCR; LatexOCR(None)"

# Hugging Face Spaces expects the app to listen on port 7860 by default;
# Render injects its own $PORT (handled by the CMD below).
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/healthz || exit 1

# gunicorn with a single worker + several threads: Pix2Tex/torch models are
# memory-heavy, so we avoid multiple worker *processes* (which would each
# load a full copy of the model) and rely on threads for concurrency instead.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 120 app:app"]
