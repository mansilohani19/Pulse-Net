# app/main.py
# FastAPI app: POST /predict scores a flow, /ws/alerts streams events to dashboards.
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.inference import Detector

STATIC = Path(__file__).parent / "static"


class FlowRequest(BaseModel):
    flow: Dict[str, Any]                 # the 41 raw NSL-KDD features
    true_label: Optional[str] = None     # only for the demo display; the model never uses it


class ConnectionManager:
    """Keeps the list of open dashboard connections and broadcasts to all of them."""

    def __init__(self):
        self.clients: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.clients:
            self.clients.remove(ws)

    async def broadcast(self, message: dict):
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:            # client went away
                self.disconnect(ws)


manager = ConnectionManager()
stats = {"flows": 0, "alerts": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = Detector()      # load the model once, not per request
    yield


app = FastAPI(title="PulseNet", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "threshold": app.state.detector.threshold, "clients": len(manager.clients)}


@app.get("/stats")
def get_stats():
    return stats


@app.post("/predict")
async def predict(req: FlowRequest):
    det: Detector = app.state.detector
    missing = det.missing_features(req.flow)
    if missing:
        raise HTTPException(422, f"missing features: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    t0 = time.perf_counter()
    score, is_anomaly = det.score(req.flow)
    latency_ms = (time.perf_counter() - t0) * 1000

    stats["flows"] += 1
    stats["alerts"] += int(is_anomaly)
    event = {
        "id": stats["flows"], "ts": time.time(), "score": score,
        "threshold": det.threshold, "is_anomaly": is_anomaly,
        "true_label": req.true_label, "latency_ms": round(latency_ms, 2),
        "service": req.flow.get("service"), "protocol": req.flow.get("protocol_type"),
    }
    await manager.broadcast(event)
    return event


@app.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()      # keep the connection open
    except WebSocketDisconnect:
        manager.disconnect(ws)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
