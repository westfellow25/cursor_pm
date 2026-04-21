"""Pulse — AI-powered Product Intelligence Platform.

Main application entry point. Run with:
    uvicorn pulse.main:app --reload
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pulse.api.routes import router
from pulse.config import settings
from pulse.database import init_db

# Register connectors
import pulse.connectors.csv_connector  # noqa: F401
import pulse.connectors.intercom  # noqa: F401
import pulse.connectors.slack  # noqa: F401
import pulse.connectors.api_connector  # noqa: F401

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Pulse",
    description="AI-powered Product Intelligence Platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API
app.include_router(router, prefix="/api/v1")

# Serve frontend static files
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the React SPA for all non-API routes."""
        file_path = FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIST / "index.html"))
else:
    @app.get("/")
    async def root():
        return {
            "name": "Pulse",
            "version": "0.1.0",
            "description": "AI-powered Product Intelligence Platform",
            "docs": "/docs",
            "status": "Frontend not built. Run 'cd frontend && npm run build' first.",
        }


@app.on_event("startup")
def on_startup():
    """Initialise database on first run."""
    init_db()
    logging.getLogger(__name__).info("Pulse platform started — %s", settings.database_url)
