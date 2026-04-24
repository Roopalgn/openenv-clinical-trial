# Chapter 15: Deployment — Sharing Your Work

## From Training to Production

You've trained your agent. It can design clinical trials. Now what? You need to **deploy** it — make it accessible to users.

Our project deploys as a **Docker container on HuggingFace Spaces**, providing both a web API and a visual dashboard.

## Docker: Packaging Your Application

### What's Docker?

Docker packages your entire application — code, libraries, Python version, everything — into a single "container" that runs identically on any machine.

**Analogy:** A Docker container is like a shipping container at a port. The crane doesn't care what's inside — it handles every container the same way. Whether it's running on your laptop, a cloud server, or an H100 machine, the container works identically.

### Our Dockerfile

```dockerfile
# From Dockerfile (simplified)

# Start with a base image that has Python
FROM python:3.11-slim

# Install our dependencies
COPY pyproject.toml .
RUN pip install .

# Copy our code
COPY . .

# Start the FastAPI server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

This creates a container that:
1. Has Python 3.11
2. Has all our dependencies (FastAPI, scipy, pydantic, numpy, etc.)
3. Starts the web server on port 8000

### The Entrypoint Script

```bash
# From entrypoint.sh
#!/bin/bash
set -e

# Start the FastAPI server
exec uvicorn server.app:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --workers 1
```

## The FastAPI Web Server

Our server provides these endpoints:

```
GET  /ping    → Health check ("Is the server alive?")
POST /reset   → Start a new episode
POST /step    → Take one action
GET  /state   → Get current episode metadata
GET  /schema  → Get JSON schemas for action/observation
WS   /ws      → WebSocket for real-time interaction
GET  /dashboard → Visual dashboard
```

### API Example (Using curl or Python)

```python
import requests

# Health check
r = requests.get("http://localhost:8000/ping")
# {"status": "ok"}

# Start a new episode
r = requests.post("http://localhost:8000/reset", json={"seed": 42})
observation = r.json()  # TrialObservation

# Take an action
action = {
    "action_type": "set_primary_endpoint",
    "parameters": {"endpoint": "PFS"},
    "justification": "Standard for oncology",
    "confidence": 0.8
}
r = requests.post("http://localhost:8000/step", json=action)
result = r.json()
# {"observation": {...}, "reward": {...}, "done": false, "info": {...}}
```

### The WebSocket Endpoint

For real-time interaction, we provide a WebSocket endpoint:

```python
# From server/app.py
@app.websocket("/ws")
async def websocket_step(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            action = TrialAction(**data)
            obs, reward, done, info = _manager.step(action)
            await ws.send_json({
                "observation": obs.model_dump(),
                "reward": reward.model_dump(),
                "done": done,
                "info": info,
            })
            if done:
                break
    except WebSocketDisconnect:
        pass
```

WebSockets allow continuous, low-latency back-and-forth communication — perfect for step-by-step trial interaction.

## HuggingFace Spaces

### What Are HuggingFace Spaces?

HuggingFace Spaces is a free hosting platform for AI demos. You push your Docker container, and they run it with a public URL.

### The OpenEnv Configuration

```yaml
# From openenv.yaml
name: clinical-trial-designer
version: 0.2.1
type: environment
runtime: docker
```

This tells the OpenEnv framework how to discover and interact with our environment.

### The Dashboard

Our project includes a live dashboard (`server/dashboard.py` + `dashboard.html`) that shows:
- Recent episode results
- Reward statistics
- Live WebSocket streaming of ongoing episodes
- Episode transcript replay

```python
# From server/dashboard.py
router = APIRouter()

@router.get("/dashboard/stats")
def dashboard_stats():
    """Return aggregate statistics for the dashboard."""
    return {
        "total_episodes": ...,
        "mean_reward": ...,
        "success_rate": ...,
        "current_tier": ...,
    }

@router.get("/dashboard/recent")
def dashboard_recent():
    """Return most recent episode summaries."""
    ...
```

## The Global Exception Handler

For production reliability, we catch ALL unhandled exceptions:

```python
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch all unhandled exceptions, log full stack trace, return 500 JSON."""
    logger.error("Unhandled exception on %s %s\n%s",
                 request.method, request.url.path, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )
```

This ensures:
1. The server never crashes silently
2. Full stack traces are logged for debugging
3. Users get a clean error message (not raw Python tracebacks)

## Testing: 249 Passing Tests

Our project has comprehensive test coverage:

```
tests/
├── test_adversarial_designer.py       # AdversarialDesigner behavior
├── test_config.py                      # Configuration loading
├── test_curriculum_controller.py       # Curriculum advancement logic
├── test_episode_logger_wiring.py       # Logging pipeline
├── test_episode_manager_compliance.py  # FDA compliance in EpisodeManager
├── test_exception_handling.py          # Error handling
├── test_integration.py                 # Full environment integration
├── test_judge.py                       # Trial judge behavior
├── test_noise_model.py                 # Domain randomization
├── test_output_generator.py            # Observation generation
├── test_phase_detector.py              # Phase classification
├── test_smoke.py                       # Quick sanity checks
├── test_trial_simulator.py             # Trial simulation
```

**Why so many tests?** Each component is tested independently. When something breaks, the test tells you EXACTLY which component failed and why. This is critical for a complex system with many interacting parts.

> **Design Decision Box: Testing Philosophy**
>
> We test **every component in isolation** AND **the full system together**:
> - **Unit tests:** Does `calculate_power(0.5, 200, 0.05)` return 0.80? (Yes/No)
> - **Integration tests:** Does `env.reset() → step() → step() → ...` produce valid rewards?
> - **Smoke tests:** Does the server start without crashing?
>
> This catches bugs at three levels:
> 1. "The power formula has a bug" → unit test catches it
> 2. "Power calculation works but reward computer uses it wrong" → integration test catches it
> 3. "Everything works but the Docker container won't start" → smoke test catches it

---

## Chapter 15 Glossary

| Keyword | Definition |
|---------|-----------|
| **Deployment** | Making your application accessible to users |
| **Docker** | Platform for packaging applications in portable containers |
| **Container** | A packaged application with all its dependencies |
| **Dockerfile** | Instructions for building a Docker container |
| **FastAPI** | Modern Python web framework for building APIs |
| **Endpoint/Route** | A URL path that the server responds to (e.g., /reset) |
| **REST API** | An API that uses HTTP methods (GET, POST) to interact |
| **WebSocket** | Protocol for real-time, two-way communication over a single connection |
| **HuggingFace Spaces** | Free hosting platform for AI demos |
| **uvicorn** | ASGI server that runs FastAPI applications |
| **Unit Test** | Test of a single function or component in isolation |
| **Integration Test** | Test of multiple components working together |
| **Smoke Test** | Quick test that the application starts without errors |
