# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Victor AI Coding Assistant - Docker Image
# Supports both air-gapped (local LLM) and cloud provider modes

# Stage 1: Build Rust native extensions
FROM rust:1.75-slim-bookworm AS rust-builder

WORKDIR /build

# Install dependencies for PyO3
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy Rust source (at root level in this project)
COPY Cargo.toml Cargo.lock* ./
COPY src/ ./src/
COPY pyproject.toml ./

# Build Rust extensions
RUN pip3 install --break-system-packages maturin && \
    maturin build --release --out dist

# Stage 2: Python runtime
FROM python:3.12-slim-bookworm AS runtime

# Labels
LABEL org.opencontainers.image.title="Victor AI Coding Assistant"
LABEL org.opencontainers.image.description="Enterprise AI coding assistant with 21 LLM providers"
LABEL org.opencontainers.image.authors="Vijaykumar Singh <vijay@anvaiops.com>"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/vjsingh/victor"

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VICTOR_HOME=/home/victor \
    VICTOR_DATA=/data

# Create non-root user
RUN groupadd --gid 1000 victor && \
    useradd --uid 1000 --gid victor --shell /bin/bash --create-home victor

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create data directory
RUN mkdir -p ${VICTOR_DATA} && chown victor:victor ${VICTOR_DATA}

WORKDIR /app

# Copy built Rust wheel from builder
COPY --from=rust-builder /build/dist/*.whl /tmp/

# Copy application code
COPY --chown=victor:victor pyproject.toml README.md ./
COPY --chown=victor:victor victor/ ./victor/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install /tmp/*.whl && \
    pip install -e "." && \
    rm -rf /tmp/*.whl

# Download embedding model for air-gapped mode (optional, adds ~500MB)
ARG DOWNLOAD_EMBEDDING_MODEL=true
RUN if [ "$DOWNLOAD_EMBEDDING_MODEL" = "true" ]; then \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"; \
    fi

# Switch to non-root user
USER victor

# Create victor config directory
RUN mkdir -p ~/.victor

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD victor --version || exit 1

# Default command
ENTRYPOINT ["victor"]
CMD ["--help"]

# Expose API server port
EXPOSE 8765
