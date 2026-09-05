# Production Dockerfile for Chandrayaan-2 Optical Image Correspondence
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose Streamlit default port
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start the Streamlit application
ENTRYPOINT ["streamlit", "run", "lunar_correspondence/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

