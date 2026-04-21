FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install Node for frontend build
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Frontend build
COPY frontend/ frontend/
RUN cd frontend && npm ci && npm run build

# Backend code
COPY pulse/ pulse/
COPY scripts/ scripts/
COPY pyproject.toml .

EXPOSE 8000

# Seed data + start server
CMD ["sh", "-c", "python -m scripts.seed 2>/dev/null; uvicorn pulse.main:app --host 0.0.0.0 --port 8000"]
