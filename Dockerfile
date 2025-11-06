# Start from official Python slim image
FROM python:3.11-slim

# Set a working directory
WORKDIR /app

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages required for headless OpenCV and general ML libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    wget \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Expose port
EXPOSE 8000

# Run as non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
