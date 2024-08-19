FROM python:3.10-slim AS base

ENV LANG C.UTF-8
ENV PYTHONUNBUFFERED 1

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tzdata \
    git \
    libatlas-base-dev\
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir poetry==1.8.3 setuptools && poetry config virtualenvs.create false


RUN pip install --upgrade pip && pip install --no-cache-dir poetry gunicorn uvicorn && poetry config virtualenvs.create false


FROM base AS install

WORKDIR /app
COPY pyproject.toml poetry.lock /app/
RUN poetry install --no-dev

FROM install AS runtime
COPY . /app/

ENV PORT=7999

ENTRYPOINT ["sh", "-c", "uvicorn project.api:app --host 0.0.0.0 --port $PORT --reload"]
