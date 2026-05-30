from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query


app = FastAPI(title="Mock trace log API", version="0.1.0")

TelemetryMap = dict[str, dict[str, object]]


def load_telemetry() -> tuple[TelemetryMap, TelemetryMap]:
    telemetry_path = Path(__file__).with_name("telemetry.json")
    payload = json.loads(telemetry_path.read_text(encoding="utf-8"))
    return payload["logs"], payload["monitoring"]


LOGS, MONITORING_RESULTS = load_telemetry()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/logs/{trace_id}")
def get_logs(trace_id: str) -> dict[str, object]:
    try:
        return LOGS[trace_id]
    except KeyError as error:
        raise HTTPException(status_code=404, detail="trace_id not found") from error


@app.get("/api/monitoring")
def query_monitoring(query: str = Query(..., min_length=1)) -> dict[str, object]:
    try:
        return MONITORING_RESULTS[query]
    except KeyError as error:
        raise HTTPException(
            status_code=404, detail="monitoring query not found"
        ) from error
