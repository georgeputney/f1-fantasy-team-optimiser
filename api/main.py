"""FastAPI service backing the Pitwall frontend - thin wrapper over app/, no pipeline logic here.

Run for local dev: uvicorn api.main:app --reload --port 8000
In production this also serves the built frontend (web/dist) as static files, so a single
service (this one) is what gets deployed - see web/dist mount below.
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.ladder import build_ladder
from api.team import build_team
from api.breakdown import build_breakdown
from api.value import build_value
from api.track_record import build_track_record

app = FastAPI(title="Pitwall API")

# Vite dev server origin - only relevant when running the frontend and API as two dev processes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/ladder")
def ladder(
    budget: Optional[float] = None,
    squad_mode: str = "model",
    drivers: Optional[str] = None,
    constructors: Optional[str] = None,
    free_transfers: int = 2,
):
    return build_ladder(
        budget=budget, squad_mode=squad_mode,
        drivers=drivers.split(",") if drivers else [],
        constructors=constructors.split(",") if constructors else [],
        free_transfers=free_transfers,
    )


@app.get("/api/team")
def team(
    budget: Optional[float] = None,
    squad_mode: str = "model",
    drivers: Optional[str] = None,
    constructors: Optional[str] = None,
    free_transfers: int = 2,
):
    return build_team(
        budget=budget, squad_mode=squad_mode,
        drivers=drivers.split(",") if drivers else [],
        constructors=constructors.split(",") if constructors else [],
        free_transfers=free_transfers,
    )


@app.get("/api/breakdown")
def breakdown(
    season: Optional[int] = None,
    round: Optional[int] = None,
    budget: Optional[float] = None,
    squad_mode: str = "model",
    drivers: Optional[str] = None,
    constructors: Optional[str] = None,
    free_transfers: int = 2,
):
    return build_breakdown(
        season=season, round_num=round,
        budget=budget, squad_mode=squad_mode,
        drivers=drivers.split(",") if drivers else [],
        constructors=constructors.split(",") if constructors else [],
        free_transfers=free_transfers,
    )


@app.get("/api/value")
def value(
    budget: Optional[float] = None,
    squad_mode: str = "model",
    drivers: Optional[str] = None,
    constructors: Optional[str] = None,
    free_transfers: int = 2,
):
    return build_value(
        budget=budget, squad_mode=squad_mode,
        drivers=drivers.split(",") if drivers else [],
        constructors=constructors.split(",") if constructors else [],
        free_transfers=free_transfers,
    )


@app.get("/api/track-record")
def track_record():
    return build_track_record()


# production: FastAPI serves the built React app directly, so Render only needs one web service
WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")