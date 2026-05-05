FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ python3-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
COPY requirements/base.txt requirements/api.txt ./requirements/
RUN pip install --no-cache-dir -r requirements/api.txt
COPY . .
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
