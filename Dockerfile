FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (jika diperlukan oleh TensorFlow dkk)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt slowapi

# Copy application source code (API dan Model)
COPY api/ /app/api/
COPY models/ /app/models/

# Environment Variable untuk port dan Gemini API (harus di-inject saat run)
ENV PORT=8000
ENV GEMINI_API_KEY=""

# Expose port untuk akses eksternal
EXPOSE 8000

# Jalankan Uvicorn Server dari folder root (supaya path models cocok)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
