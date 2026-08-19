FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.8.3

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

# Copy project source
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY dashboard/ ./dashboard/

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "marketplace_trust.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
