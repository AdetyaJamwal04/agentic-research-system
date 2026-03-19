FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data/documents

EXPOSE 8000

# Use PORT env var (Render sets this), fallback to 8000
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
