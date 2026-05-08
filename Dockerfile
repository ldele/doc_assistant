FROM python:3.12-slim

# System dependencies needed by some Python packages
# - build-essential: compiles C extensions
# - libxml2-dev, libxslt1-dev: lxml's native deps
# - curl: useful for healthchecks and debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first, install separately.
# Dependencies only reinstall when pyproject.toml changes
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Copy the rest of the application
COPY apps/ ./apps/
COPY scripts/ ./scripts/

# Chainlit listens on 8000 by default
EXPOSE 8000

# --host 0.0.0.0 makes it accessible from outside the container.
CMD ["chainlit", "run", "apps/chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]