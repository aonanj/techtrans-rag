# Use a slim Python image (matches local development Python 3.13 artifacts)
FROM python:3.13-slim AS builder

# Metadata
LABEL maintainer="techtrans <no-reply@example.com>"

# Environment - Optimized for Cloud Run and Cloud Logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080 \
    LOG_TO_CLOUD=true \
    GOOGLE_CLOUD_PROJECT="tech-trans-rag" \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies required by some Python packages (Tesseract for OCR, libs for images/PDFs)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    wget \
    ca-certificates \
    git \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies (include gunicorn for production)
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080 \
    LOG_TO_CLOUD=true \
    GOOGLE_CLOUD_PROJECT="tech-trans-rag"

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    wget \
    ca-certificates \
    git \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*

COPY . .

CMD ["bash", "-lc", "exec gunicorn 'app:create_app()' --bind 0.0.0.0:${PORT} --workers ${WEB_CONCURRENCY:-2} --threads ${GUNICORN_THREADS:-8} --timeout ${GUNICORN_TIMEOUT:-120} --access-logfile - --error-logfile -"]