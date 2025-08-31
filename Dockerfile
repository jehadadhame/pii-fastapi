# -------------------------------
#  Dockerfile for FastAPI + Conda
# -------------------------------
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment.yml
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "pii-env", "/bin/bash", "-c"]

# Install uvicorn and fastapi inside the conda env
RUN pip install fastapi uvicorn python-multipart


# Copy application code
COPY main.py .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app with uvicorn inside conda env
CMD ["conda", "run", "--no-capture-output", "-n", "pii-env", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
