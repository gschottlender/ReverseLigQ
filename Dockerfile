# -------------------------------------------------------------------
# BASE IMAGE
# -------------------------------------------------------------------
FROM continuumio/miniconda3:latest

# -------------------------------------------------------------------
# SYSTEM DEPENDENCIES
# Node.js is only used to build the React frontend during image build.
# -------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends nodejs npm && \
    rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------
# WORKDIR
# -------------------------------------------------------------------
WORKDIR /app

# -------------------------------------------------------------------
# COPY ENVIRONMENT
# -------------------------------------------------------------------
COPY environment.yml .

# -------------------------------------------------------------------
# CREATE CONDA ENVIRONMENT
# -------------------------------------------------------------------
RUN conda env create -n reverse_ligq -f environment.yml && \
    conda clean -afy && \
    rm -rf /opt/conda/pkgs

# -------------------------------------------------------------------
# ACTIVATE ENVIRONMENT
# -------------------------------------------------------------------
ENV PATH="/opt/conda/envs/reverse_ligq/bin:${PATH}"

# -------------------------------------------------------------------
# HUGGING FACE CACHE DIRECTORY
# Create a writable directory for Hugging Face / transformers cache.
# We give it 777 permissions so it works even when the container is
# run as a non-root user (e.g. with `-u $(id -u):$(id -g)`).
# -------------------------------------------------------------------
RUN mkdir -p /hf_cache && chmod 777 /hf_cache

# Set default cache locations for Hugging Face and transformers.
ENV HF_HOME=/hf_cache

# -------------------------------------------------------------------
# COPY SOURCE CODE
# -------------------------------------------------------------------
COPY . .

# -------------------------------------------------------------------
# BUILD FRONTEND
# -------------------------------------------------------------------
RUN cd web/frontend && \
    npm ci && \
    npm run build && \
    rm -rf node_modules

# -------------------------------------------------------------------
# WEB PORT
# -------------------------------------------------------------------
EXPOSE 8000

# -------------------------------------------------------------------
# DEFAULT WEB COMMAND
# -------------------------------------------------------------------
CMD ["uvicorn", "web.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]

# -------------------------------------------------------------------
# CLI USAGE
# -------------------------------------------------------------------
# To run the historical CLI instead of the web app, override the command:
# docker run ... gschottlender/reverseligq:latest python rev_ligq.py --help
