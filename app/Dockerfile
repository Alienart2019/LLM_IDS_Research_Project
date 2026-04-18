FROM python:3.12-slim

# System deps for scapy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY train.py .

# Pre-create directories so volumes mount cleanly
RUN mkdir -p models db data

# Non-root user
RUN useradd -m ids && chown -R ids:ids /app
USER ids

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
