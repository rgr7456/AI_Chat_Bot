# HRMS face + OCR service — FastAPI + OpenCV + ONNX Runtime
FROM python:3.11-slim-bookworm

# System libraries the ML stack needs:
#   libgl1, libglib2.0-0  -> OpenCV (opencv-python)
#   libgomp1              -> ONNX Runtime (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python deps first (better layer caching).
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code, migrations and the ONNX model files.
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini .
COPY models ./models

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

# Render/most PaaS inject $PORT; default to 8000 for local `docker run`.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
