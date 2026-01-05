# -------------------------------------------------------------------
# BASE IMAGE
# Use the official Miniconda3 image as a base.
# This provides conda to manage Python dependencies.
# -------------------------------------------------------------------
FROM continuumio/miniconda3:latest

# -------------------------------------------------------------------
# WORKDIR
# All subsequent commands will run inside /app.
# When building, you must run `docker build` from the root of the
# cloned ReverseLigQ repository so that the context contains the code.
# -------------------------------------------------------------------
WORKDIR /app

# -------------------------------------------------------------------
# COPY ENVIRONMENT
# Copy only the environment file first to leverage Docker layer caching.
# If environment.yml does not change, the conda environment layer
# will be reused between builds.
# -------------------------------------------------------------------
COPY environment.yml .

# -------------------------------------------------------------------
# CREATE CONDA ENVIRONMENT
# Create the conda environment used by ReverseLigQ, then clean up
# conda caches to reduce image size.
# -------------------------------------------------------------------
RUN conda env create -n reverse_ligq -f environment.yml && \
    conda clean -afy && \
    rm -rf /opt/conda/pkgs

# -------------------------------------------------------------------
# ACTIVATE ENVIRONMENT
# Add the environment's bin directory to PATH so `python` and other
# tools run from the reverse_ligq environment by default.
# -------------------------------------------------------------------
ENV PATH="/opt/conda/envs/reverse_ligq/bin:${PATH}"

# -------------------------------------------------------------------
# COPY SOURCE CODE
# Copy the entire repository into /app.
# It is strongly recommended to use a .dockerignore file to avoid
# sending unnecessary files in the build context (e.g. .git, databases, 
# large result folders, __pycache__, etc.).
# -------------------------------------------------------------------
COPY . .

# -------------------------------------------------------------------
# ENTRYPOINT
# Default entrypoint: run the ReverseLigQ CLI.
# This allows commands like:
#
#   docker run reverseligq:latest --organism 13 --query-smiles "..."
#
# which is equivalent to:
#   python rev_ligq.py --organism 13 --query-smiles "..."
# -------------------------------------------------------------------
ENTRYPOINT ["python", "rev_ligq.py"]

# -------------------------------------------------------------------
# DEFAULT CMD
# If no arguments are provided, show the help message.
# -------------------------------------------------------------------
CMD ["--help"]
