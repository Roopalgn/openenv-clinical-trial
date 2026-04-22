"""
Dashboard backend — serves live episode data for the demo dashboard.

Exposes:
  GET  /dashboard        → serves dashboard.html (or a JSON stub if not present)
  GET  /dashboard/stats  → current reward_log.csv summary as JSON
  GET  /dashboard/recent → last N episode transcript summaries
  WS   /dashboard/stream → SSE-style WebSocket streaming live step events

The dashboard is designed to be embeddable in HF Space and shows:
  - Current episode replay (action log)
  - Reward curve chart data
  - Scenario difficulty progression
  - Agent action log
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from server.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# In-memory broadcast list for live step streaming
_live_subscribers: list[WebSocket] = []


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


@router.get("", response_class=JSONResponse)
async def dashboard_index() -> Any:
    """Serve dashboard.html if present, otherwise return a JSON status stub."""
    html_path = Path("dashboard.html")
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    # Fallback: return JSON so the endpoint is always reachable
    return JSONResponse(
        content={
            "status": "dashboard_available",
            "message": "Place dashboard.html at repo root to serve the full UI.",
            "endpoints": {
                "stats": "/dashboard/stats",
                "recent": "/dashboard/recent",
                "stream": "ws://<host>/dashboard/stream",
            },
        }
    )


@router.get("/stats")
async def dashboard_stats() -> dict:
    """Return summary statistics from reward_log.csv.

    Returns:
        Dict with total_episodes, mean_reward, success_rate, and
        per-tier breakdown.
    """
    reward_csv = settings.log_path / "reward_log.csv"
    if not reward_csv.exists():
        return {
            "total_episodes": 0,
            "mean_reward": 0.0,
            "success_rate": 0.0,
            "tiers": {},
        }  # noqa: E501

    rows: list[dict] = []
    try:
        with reward_csv.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    except OSError as exc:
        logger.warning("dashboard_stats: cannot read %s: %s", reward_csv, exc)
        return {"error": str(exc)}

    if not rows:
        return {
            "total_episodes": 0,
            "mean_reward": 0.0,
            "success_rate": 0.0,
            "tiers": {},
        }

    total = len(rows)
    rewards = [float(r.get("total_reward", 0)) for r in rows]
    successes = sum(1 for r in rows if r.get("terminal_outcome") == "success")

    # Per-tier breakdown
    tier_stats: dict[str, dict] = {}
    for row in rows:
        tier = str(row.get("curriculum_tier", "0"))
        if tier not in tier_stats:
            tier_stats[tier] = {"episodes": 0, "successes": 0, "total_reward": 0.0}
        tier_stats[tier]["episodes"] += 1
        tier_stats[tier]["total_reward"] += float(row.get("total_reward", 0))
        if row.get("terminal_outcome") == "success":
            tier_stats[tier]["successes"] += 1

    for tier, stats in tier_stats.items():
        n = stats["episodes"]
        stats["mean_reward"] = stats["total_reward"] / n if n > 0 else 0.0
        stats["success_rate"] = stats["successes"] / n if n > 0 else 0.0

    return {
        "total_episodes": total,
        "mean_reward": sum(rewards) / total,
        "success_rate": successes / total,
        "reward_history": rewards[-100:],  # last 100 for chart
        "tiers": tier_stats,
    }


@router.get("/recent")
async def dashboard_recent(n: int = 10) -> list[dict]:
    """Return the last *n* episode summary records from transcript JSONL files.

    Args:
        n: Number of recent episodes to return (default 10, max 50).

    Returns:
        List of summary dicts (most recent last).
    """
    n = min(n, 50)
    transcripts_dir = settings.log_path / "episode_transcripts"
    if not transcripts_dir.exists():
        return []

    # Collect all JSONL files sorted by modification time (newest last)
    jsonl_files = sorted(
        transcripts_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
    )
    recent_files = jsonl_files[-n:]

    summaries: list[dict] = []
    for path in recent_files:
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if record.get("type") == "summary":
                        summaries.append(record)
                        break  # one summary per file
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("dashboard_recent: cannot read %s: %s", path, exc)

    return summaries[-n:]


# ---------------------------------------------------------------------------
# WebSocket live stream
# ---------------------------------------------------------------------------


@router.websocket("/stream")
async def dashboard_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming live step events to the dashboard.

    Clients connect and receive JSON messages whenever a step is broadcast
    via ``broadcast_step()``.  The connection stays open until the client
    disconnects.
    """
    await websocket.accept()
    _live_subscribers.append(websocket)
    try:
        # Keep connection alive; client sends pings or just listens
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _live_subscribers:
            _live_subscribers.remove(websocket)


async def broadcast_step(event: dict) -> None:
    """Broadcast a step event to all connected dashboard WebSocket clients.

    Called by EpisodeManager (or app routes) after each step to push live
    data to the dashboard.  Silently drops disconnected clients.

    Args:
        event: Dict with step data (action, observation, reward, phase, etc.)
    """
    disconnected: list[WebSocket] = []
    for ws in list(_live_subscribers):
        try:
            await ws.send_json(event)
        except Exception:  # noqa: BLE001
            disconnected.append(ws)
    for ws in disconnected:
        if ws in _live_subscribers:
            _live_subscribers.remove(ws)
