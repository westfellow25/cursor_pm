from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse

from .pipeline import analyze_feedback_csv, generate_discovery, load_feedback_csv
from .schemas import AnalyzeResponse, DiscoveryResponse

app = FastAPI(title="AI Product Discovery MVP", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


LAST_RUN: dict[str, str] = {"prd": "", "jira": ""}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Product Discovery MVP</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 920px; margin: 2rem auto; padding: 0 1rem; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
    button { padding: 0.5rem 0.9rem; cursor: pointer; }
    pre { white-space: pre-wrap; background: #f8f8f8; padding: 0.75rem; border-radius: 8px; }
    .links a { margin-right: 1rem; }
  </style>
</head>
<body>
  <h1>AI Product Discovery MVP</h1>
  <div class="card">
    <label for="fileInput"><strong>Upload CSV</strong></label><br /><br />
    <input id="fileInput" type="file" accept=".csv" />
    <button id="analyzeBtn">Analyze</button>
    <p id="status"></p>
    <div class="links">
      <a href="/download/prd" target="_blank" rel="noopener">Download PRD</a>
      <a href="/download/jira" target="_blank" rel="noopener">Download Jira</a>
    </div>
  </div>

  <div class="card">
    <h2>Results</h2>
    <pre id="results">No analysis yet.</pre>
  </div>

  <script>
    const analyzeBtn = document.getElementById('analyzeBtn');
    const fileInput = document.getElementById('fileInput');
    const statusEl = document.getElementById('status');
    const resultsEl = document.getElementById('results');

    analyzeBtn.addEventListener('click', async () => {
      const file = fileInput.files[0];
      if (!file) {
        statusEl.textContent = 'Please choose a CSV file first.';
        return;
      }

      const formData = new FormData();
      formData.append('file', file);

      statusEl.textContent = 'Analyzing...';
      resultsEl.textContent = '';

      try {
        const response = await fetch('/analyze', {
          method: 'POST',
          body: formData,
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || 'Request failed');
        }
        statusEl.textContent = 'Analysis complete.';
        resultsEl.textContent = JSON.stringify(data, null, 2);
      } catch (error) {
        statusEl.textContent = `Error: ${error.message}`;
        resultsEl.textContent = 'Failed to analyze CSV.';
      }
    });
  </script>
</body>
</html>
"""


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
