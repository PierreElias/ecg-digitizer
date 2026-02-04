# Hugging Face Spaces Dockerfile for ECG Digitizer
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Open-ECG-Digitizer source
COPY ../Open-ECG-Digitizer /app/Open-ECG-Digitizer

# Copy web app
COPY . /app/web
WORKDIR /app/web

# Expose port (Hugging Face uses 7860)
EXPOSE 7860

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "app:app"]
