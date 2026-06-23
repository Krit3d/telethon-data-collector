FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ python3-dev libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip
COPY requirements/base.txt requirements/scraper.txt ./requirements/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary -r requirements/scraper.txt
COPY . .
CMD ["python", "-m", "src.parser.parser"]
