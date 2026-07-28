FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app
COPY pyproject.toml README.md ./
COPY src ./src
COPY migrations ./migrations
COPY scripts ./scripts
COPY web ./web
RUN pip install --no-cache-dir ".[storage]"

USER app
EXPOSE 8000
CMD ["uvicorn", "plagiarism_detection.api:app", "--host", "0.0.0.0", "--port", "8000"]
