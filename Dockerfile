FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Upgrade pip & install dependencies
RUN pip install --upgrade pip wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port for Cloud Run
EXPOSE 8080

# Start server
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
