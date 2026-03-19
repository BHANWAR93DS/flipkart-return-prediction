# ============================================================
# Dockerfile — Flipkart Return Prediction API
# ============================================================

# Base image — Python 3.11 slim (chhota size)
FROM python:3.11-slim

# Who made this
LABEL maintainer="MLOps Engineer"
LABEL version="1.0.0"
LABEL description="Flipkart Return Prediction API"

# Set working directory inside container
WORKDIR /app

# WHY copy requirements first?
# Docker caches layers — agar sirf code change ho toh
# packages dobara install nahi honge. Faster builds!
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY config/ ./config/
COPY src/     ./src/
COPY api/     ./api/
COPY models/  ./models/
COPY setup.py .

# Install project as package
RUN pip install -e . --no-build-isolation

# Create necessary folders
RUN mkdir -p logs data/raw data/processed data/reference mlruns

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check — Docker will ping this every 30 seconds
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]