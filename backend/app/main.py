from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from app.pipeline import analyze_feedback_csv, generate_discovery, load_feedback_csv
from app.schemas import AnalyzeResponse, DiscoveryResponse

app = FastAPI(title="AI Product Discovery MVP", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


LAST_RUN: dict[str, str] = {"prd": "", "jira": ""}


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


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    content = await file.read()
    try:
        analysis = analyze_feedback_csv(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to analyze CSV: {exc}") from exc

    LAST_RUN["prd"] = analysis.prd_text
    LAST_RUN["jira"] = analysis.jira_tickets_text
    return analysis


@app.get("/download/prd")
def download_prd() -> PlainTextResponse:
    if not LAST_RUN["prd"]:
        raise HTTPException(status_code=404, detail="No PRD available. Run /analyze first.")
    return PlainTextResponse(
        content=LAST_RUN["prd"],
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="PRD.md"'},
    )


@app.get("/download/jira")
def download_jira() -> PlainTextResponse:
    if not LAST_RUN["jira"]:
        raise HTTPException(status_code=404, detail="No Jira tickets available. Run /analyze first.")
    return PlainTextResponse(
        content=LAST_RUN["jira"],
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="jira_tickets.md"'},
    )
