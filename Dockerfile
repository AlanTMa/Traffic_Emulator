# One image for every role; docker-compose.yml picks the command.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# default: the single-process emulator
CMD ["python", "-m", "src.cli", "run", "--config", "config/paper_5x3.yaml", "--output-dir", "runs/reference"]
