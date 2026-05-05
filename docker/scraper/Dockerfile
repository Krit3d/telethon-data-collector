FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ python3-dev libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
COPY requirements/base.txt requirements/scraper.txt ./requirements/
RUN pip install --no-cache-dir -r requirements/scraper.txt
COPY . .
CMD ["python", "-m", "src.parser.parser"]
