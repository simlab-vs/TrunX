FROM python:3.12-slim
# FROM nvidia/cuda:12.6.2-base-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	UV_COMPILE_BYTECODE=1 \
	UV_LINK_MODE=copy \
    UV_NO_DEV=1

# Set working directory 
WORKDIR /app

# System dependencies for scientific Python stack and rpy2.
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
    git \
    python3-pip \
    python3-dev \ 
	curl \
	libopenblas-dev \
	liblapack-dev \
	libgomp1 \
	libgl1 \
	libglib2.0-0 \
	r-base \
    r-base-dev \
    python3-rpy2 \  
	&& rm -rf /var/lib/apt/lists/*

# Install required R packages 
RUN R -e "install.packages(c('readxl', 'dplyr', 'tidyr', 'ggplot2'), repos='https://cran.rstudio.com/')"

# Install uv.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy pyproject.toml (if you have one in TrunkX root)
COPY pyproject.toml README.md ./ 

# Copy your source code
COPY src ./src

# Copy config.yaml 
COPY config.yaml ./ 

#Copy R implementation
COPY models/r3PG ./models/r3PG

# Install dependencies
RUN uv sync

ENV PATH="/app/.venv/bin:${PATH}"

# Set R environment variables
ENV R_HOME=/usr/lib/R 
ENV LD_LIBRARY_PATH=/usr/lib/R/lib

# Run necessary scripts
ENTRYPOINT ["uv", "run", "python"]
CMD ["/app/src/trunx/gp3/jax_morris_loglikelihood.py"]
