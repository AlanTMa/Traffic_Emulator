# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY tests/ ./tests/

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Default command: run the emulator with the baseline config
CMD ["python", "src/cli.py", "run", "--config", "config/paper_5x3.yaml"]
