# Use an official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies
# (needed for pdfminer.six and PyTorch wheels sometimes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pip and upgrade
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements (create your own requirements.txt if needed)
# If you already have requirements.yml, you can export to txt with: 
#   pip freeze > requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install your framework versions explicitly (to match your env)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pdfminer.six \
    "transformers==4.44.2" \
    "torch==2.2.2" \
    "tokenizers==0.19.1" \
    "datasets==3.0.0" \
    -f https://download.pytorch.org/whl/torch_stable.html

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
