# Use an official Python runtime as a parent image
FROM continuumio/miniconda3:4.11.0
RUN conda install -c conda-forge -v -y gcc

# Set the working directory in the container
WORKDIR /app

# Add Conda to PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Install dependencies from Conda .yml file
COPY environment.yml .
RUN conda env create -f environment.yml

# Copy the current directory contents into the container at /app
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV NAME reverse_ligq

# Activate the environment and run the app:
CMD ["conda", "run", "-n", "reverse_ligq", "streamlit", "run", "rev_lq.py", "--server.port=8501", "--server.address=0.0.0.0"]

