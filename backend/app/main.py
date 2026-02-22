from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline import generate_discovery, load_feedback_csv
from app.schemas import DiscoveryResponse

app = FastAPI(title="AI Product Discovery MVP", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/discover", response_model=DiscoveryResponse)
async def discover(file: UploadFile = File(...)) -> DiscoveryResponse:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    content = await file.read()
    try:
        df = load_feedback_csv(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV does not contain usable feedback rows")

    try:
        return generate_discovery(df)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
