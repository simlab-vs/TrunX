# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# base: system dependencies shared by the dev container and the runtime image.
#
# R packages are installed as pre-built binaries via the CRAN apt repository
# (current R) and r2u (binary CRAN packages), which avoids compiling r3PG and
# friends from source on Ubuntu.
#
# CUDA is provided by the jax[cuda12] pip wheels, so no CUDA base image is
# needed; run with `docker run --gpus all` (requires nvidia-container-toolkit).
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    # Expose the host GPU when run with --gpus (no-op on CPU-only hosts).
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
    # CRAN apt repository: current R release for Ubuntu.
    && curl -fsSL https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
        -o /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc \
    && echo "deb [arch=amd64] https://cloud.r-project.org/bin/linux/ubuntu noble-cran40/" \
        > /etc/apt/sources.list.d/cran.list \
    # r2u: all CRAN packages as binary .deb (no source compilation).
    && curl -fsSL https://eddelbuettel.github.io/r2u/assets/dirk_eddelbuettel_key.asc \
        -o /etc/apt/trusted.gpg.d/cranapt_key.asc \
    && echo "deb [arch=amd64] https://r2u.stat.illinois.edu/ubuntu noble main" \
        > /etc/apt/sources.list.d/cranapt.list \
    && printf 'Package: *\nPin: release o=CRAN-Apt Project\nPin: release l=CRAN-Apt Packages\nPin-Priority: 700\n' \
        > /etc/apt/preferences.d/99cranapt \
    && apt-get update && apt-get install -y --no-install-recommends \
        # R itself; r-base-dev provides the headers and compilers rpy2 needs.
        r-base-core \
        r-base-dev \
        # Binary R packages used by models/r3PG scripts.
        r-cran-r3pg \
        r-cran-readxl \
        r-cran-dplyr \
        r-cran-tidyr \
        r-cran-ggplot2 \
        # Runtime library for the Python eccodes/cfgrib bindings.
        libeccodes0 \
        libeccodes-data \
    && rm -rf /var/lib/apt/lists/*

# uv, plus a shared interpreter location readable by every user.
COPY --from=ghcr.io/astral-sh/uv:0.9.25 /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_LINK_MODE=copy
RUN uv python install 3.13

# ---------------------------------------------------------------------------
# dev: used by .devcontainer/devcontainer.json. Python dependencies are
# installed post-create by `uv sync` into the mounted workspace.
# ---------------------------------------------------------------------------
FROM base AS dev

RUN chown -R ubuntu:ubuntu /opt/uv

# ---------------------------------------------------------------------------
# runtime: self-contained image with the project and its locked dependencies.
# ---------------------------------------------------------------------------
FROM base AS runtime

ENV UV_COMPILE_BYTECODE=1 \
    UV_NO_DEV=1

WORKDIR /app

# Install dependencies first so they are cached independently of the source.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY config.yaml ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

ENV PATH="/app/.venv/bin:${PATH}"

CMD ["python"]
