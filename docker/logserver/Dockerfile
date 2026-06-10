FROM python:3.11-alpine

WORKDIR /app

COPY docker/logserver/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docker/logserver/app.py .
COPY dataset/gitea_telemetry.json dataset/gitea_telemetry.json

ENV LOGSERVER_TELEMETRY_FILE=/app/dataset/gitea_telemetry.json

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
