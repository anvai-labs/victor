# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
# Canonical build: --target core | mcp | native | full (default).
# See docs/development/dependencies.md for lock regeneration and image validation.
ARG PYTHON_IMAGE=python:3.12-slim-trixie@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea
ARG RUST_IMAGE=rust:1.98-slim-trixie@sha256:bce1476d4be4d78b83705bc5f428b86d640eeeea33e9dadafbc037b5703a53bf

FROM ${PYTHON_IMAGE} AS python-base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
# Git and SSH are runtime tool capabilities. Keep them even when an upstream
# advisory blocks release; the image scan decides whether publication is allowed.
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends git openssh-client ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --upgrade 'pip>=26.2.1' 'setuptools>=83.0.0'

FROM python-base AS wheels
WORKDIR /build
COPY pyproject.toml README.md LICENSE VERSION ./
COPY victor ./victor
COPY victor-contracts ./victor-contracts
RUN python -m pip wheel --no-deps --wheel-dir /wheels ./victor-contracts .

FROM python-base AS core-deps
COPY requirements.txt /locks/core.txt
# Resolve the in-repo SDK candidate before its independent PyPI release.
COPY --from=wheels /wheels /wheels
RUN python -m venv /opt/victor && \
    /opt/victor/bin/python -m pip install --upgrade 'pip>=26.2.1' && \
    /opt/victor/bin/python -m pip install --find-links=/wheels -r /locks/core.txt && \
    /opt/victor/bin/python -m pip check

FROM core-deps AS core-env
COPY --from=wheels /wheels /wheels
RUN /opt/victor/bin/python -m pip install --no-deps --force-reinstall /wheels/*.whl && \
    /opt/victor/bin/python -m pip check && \
    /opt/victor/bin/python -m pip uninstall -y pip

FROM python-base AS runtime
LABEL org.opencontainers.image.source="https://github.com/anvai-labs/victor" \
      org.opencontainers.image.licenses="Apache-2.0"
ENV PATH=/opt/victor/bin:$PATH VICTOR_HOME=/home/victor/.victor \
    HF_HOME=/home/victor/.cache/huggingface
RUN useradd -m -u 1000 -s /bin/bash victor && \
    mkdir -p /workspace /home/victor/.victor /home/victor/.cache && \
    chown -R victor:victor /workspace /home/victor && \
    python -m pip uninstall -y setuptools pip
WORKDIR /workspace
USER victor
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD victor --version || exit 1

FROM runtime AS mcp
COPY --from=core-env /opt/victor /opt/victor
ENTRYPOINT ["victor", "mcp"]
CMD ["--log-level", "WARNING"]

FROM core-deps AS api-deps
COPY requirements/api/requirements.txt /locks/api.txt
RUN /opt/victor/bin/python -m pip install -r /locks/api.txt && \
    /opt/victor/bin/python -m pip check

FROM api-deps AS api-env
COPY --from=wheels /wheels /wheels
RUN /opt/victor/bin/python -m pip install --no-deps --force-reinstall /wheels/*.whl && \
    /opt/victor/bin/python -m pip check && \
    /opt/victor/bin/python -m pip uninstall -y pip

FROM runtime AS core
COPY --from=api-env /opt/victor /opt/victor
EXPOSE 8765
CMD ["victor", "serve", "--host", "0.0.0.0", "--port", "8765"]

FROM ${RUST_IMAGE} AS rust-toolchain
FROM python-base AS native-env
COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup
ENV PATH=/usr/local/cargo/bin:$PATH RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/* && python -m pip install 'maturin>=1.10,<2'
COPY rust /build/rust
RUN cd /build/rust && maturin build --release --locked --out /native-wheels

FROM api-env AS native-install
COPY --from=native-env /native-wheels /native-wheels
RUN python -m pip --python /opt/victor install --no-deps /native-wheels/*.whl && \
    /opt/victor/bin/python -c 'import victor_native' && \
    python -m pip --python /opt/victor check

FROM runtime AS native
COPY --from=native-install /opt/victor /opt/victor
ENTRYPOINT ["victor"]
CMD ["--help"]

FROM core-deps AS embeddings-deps
COPY requirements/embeddings-cpu/requirements.txt /locks/embeddings-cpu.txt
# The lock selects CPU wheels explicitly; no CUDA, torchvision or torchaudio is
# needed by Victor's text embedding path.
RUN /opt/victor/bin/python -m pip install -r /locks/embeddings-cpu.txt && \
    /opt/victor/bin/python -m pip check
ENV HF_HOME=/model-cache/huggingface
RUN /opt/victor/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

FROM embeddings-deps AS embeddings-env
COPY --from=wheels /wheels /wheels
RUN /opt/victor/bin/python -m pip install --no-deps --force-reinstall /wheels/*.whl && \
    /opt/victor/bin/python -m pip check && \
    /opt/victor/bin/python -m pip uninstall -y pip

FROM runtime AS full
USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*
USER victor
COPY docker /app/docker
COPY examples /app/examples
COPY --from=embeddings-env /opt/victor /opt/victor
COPY --from=embeddings-env --chown=victor:victor /model-cache/huggingface /home/victor/.cache/huggingface
# Embeddings are available offline. Provider calls and network tools still need
# their configured endpoints; derived tool caches are rebuilt without migration.
CMD ["bash"]
