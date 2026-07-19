FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY backend/pyproject.toml ./pyproject.toml
COPY README.md /README.md
COPY LICENSE /LICENSE
COPY backend/app ./app

RUN pip install --no-cache-dir . \
    && mkdir -p /app/data/sqlite /app/data/uploads /app/data/chroma \
    && chown -R app:app /app

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
