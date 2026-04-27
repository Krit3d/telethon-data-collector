FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app

# Install system dependencies (need for building some packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install only torch-cpu (smaller size)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "src.parser.parser"]
