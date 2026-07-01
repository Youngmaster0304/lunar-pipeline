FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY web/requirements.prod.txt .
RUN pip install --no-cache-dir -r requirements.prod.txt

COPY . .

ENV PORT=7860
EXPOSE 7860

RUN chmod +x scripts/*.sh

# Default: web server (override CMD for Celery worker with scripts/start-worker.sh)
CMD ["bash", "scripts/start.sh"]
