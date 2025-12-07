FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    wgrib2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/
COPY models/ models/
COPY migrations/ migrations/
COPY alembic.ini .

# Install project in editable mode (for entry points)
RUN pip install --no-cache-dir -e .

# Default command (overridden by docker-compose)
CMD ["python", "--version"]
