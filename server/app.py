"""
FastAPI application for the Clinical Trial Designer environment.

Routes:
  GET  /ping   → health check
  POST /reset  → initialize episode, return TrialObservation
  POST /step   → advance episode, return StepResponse
  GET  /state  → return current TrialState
  GET  /schema → JSON schemas for TrialAction and TrialObservation
  WS   /ws     → WebSocket streaming for step interactions
"""

from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from models import TrialAction, TrialObservation, TrialState
from server.config import settings
from server.episode_manager import EpisodeManager

app = FastAPI(title="Clinical Trial Designer Environment")

_manager = EpisodeManager()


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    seed: int | None = None


class StepResponse(BaseModel):
    observation: TrialObservation
    reward: dict[str, Any]
    done: bool
    info: dict[str, Any]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/ping")
def ping() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/reset", response_model=TrialObservation)
def reset(body: ResetRequest = ResetRequest()) -> TrialObservation:
    """Initialize a new episode and return the initial observation."""
    return _manager.reset(seed=body.seed)


@app.post("/step", response_model=StepResponse)
def step(action: TrialAction) -> StepResponse:
    """Advance the episode by one step."""
    try:
        obs, reward_breakdown, done, info = _manager.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StepResponse(
        observation=obs,
        reward=reward_breakdown.model_dump(),
        done=done,
        info=info,
    )


@app.get("/state", response_model=TrialState)
def state() -> TrialState:
    """Return the current TrialState (training-loop metadata)."""
    try:
        return _manager.get_state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/schema")
def schema() -> dict[str, Any]:
    """Return JSON schemas for TrialAction and TrialObservation."""
    return {
        "TrialAction": TrialAction.model_json_schema(),
        "TrialObservation": TrialObservation.model_json_schema(),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming step interactions.

    Accepts JSON-encoded TrialAction messages and responds with StepResponse JSON.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            try:
                action = TrialAction.model_validate(data)
                obs, reward_breakdown, done, info = _manager.step(action)
                response = StepResponse(
                    observation=obs,
                    reward=reward_breakdown.model_dump(),
                    done=done,
                    info=info,
                )
                await websocket.send_json(response.model_dump())
            except RuntimeError as exc:
                await websocket.send_json({"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                await websocket.send_json({"error": f"Invalid request: {exc}"})
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "server.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
