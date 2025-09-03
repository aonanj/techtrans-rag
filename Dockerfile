# Use a slim Python image (matches local development Python 3.13 artifacts)
FROM python:3.13-slim

# Metadata
LABEL maintainer="techtrans <no-reply@example.com>"

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    GOOGLE_CLOUD_PROJECT="" \
    LOG_TO_CLOUD=true

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
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install Python dependencies (include gunicorn for production)
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir gunicorn google-cloud-logging

# Copy application code
COPY . /app

# Create a non-root user and ensure writable dirs
RUN useradd --create-home --no-log-init appuser \
 && chown -R appuser:appuser /app

USER appuser

EXPOSE $PORT

# Use Gunicorn to run the Flask app factory. Cloud Run sets $PORT; default 8080 defined above.
# We call the factory directly using the app:create_app format
# Configure gunicorn logging for Cloud Logging compatibility
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 --access-logfile - --error-logfile - --log-level info "app:create_app()"
