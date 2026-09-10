# One image for every role: controller, source, broker, dashboard and the
# single-process reference emulator. docker-compose.yml picks the command.
FROM python:3.11-slim

WORKDIR /app

# Python dependencies (numpy/scipy/pandas ship wheels; no compiler needed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and configs
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Default: the in-process reference emulator on the canonical 5x3 instance
CMD ["python", "-m", "src.cli", "run", "--config", "config/paper_5x3.yaml", "--output-dir", "runs/reference"]
