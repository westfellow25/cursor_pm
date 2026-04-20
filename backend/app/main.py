from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse

from .pipeline import analyze_feedback_csv
from .schemas import AnalyzeResponse

app = FastAPI(title="AI Product Discovery MVP", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Per-run artifact storage keyed by run_id so concurrent users don't overwrite
# each other's PRD/Jira downloads.
RUN_STORE: dict[str, dict[str, str]] = {}
RUN_ORDER: list[str] = []
MAX_RUNS = 32


def _store_run(run_id: str, prd_text: str, jira_text: str) -> None:
    RUN_STORE[run_id] = {"prd": prd_text, "jira": jira_text}
    RUN_ORDER.append(run_id)
    while len(RUN_ORDER) > MAX_RUNS:
        old = RUN_ORDER.pop(0)
        RUN_STORE.pop(old, None)


HOME_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Product Discovery MVP</title>
  <style>
    :root { color-scheme: light; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #1f2430; }
    h1 { margin-bottom: 0.25rem; }
    .subtitle { color: #6b7280; margin-top: 0; }
    .card { border: 1px solid #e4e7eb; border-radius: 12px; padding: 1.1rem 1.2rem; margin-bottom: 1rem; background: #ffffff; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
    button { padding: 0.5rem 0.95rem; cursor: pointer; background: #1f2430; color: #fff; border: 0; border-radius: 8px; font-weight: 500; }
    button:disabled { opacity: 0.5; cursor: progress; }
    input[type=file] { margin-right: 0.5rem; }
    .muted { color: #6b7280; }
    .links a { margin-right: 1rem; }
    .links a.disabled { pointer-events: none; color: #9ca3af; text-decoration: none; }
    .cluster { border: 1px solid #e4e7eb; border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.8rem; background: #fafbfc; }
    .cluster h3 { margin: 0 0 0.25rem; font-size: 1.05rem; }
    .cluster .meta { color: #6b7280; font-size: 0.88rem; margin-bottom: 0.4rem; }
    .cluster .quote { font-style: italic; color: #374151; }
    details { margin-top: 0.5rem; }
    details pre { white-space: pre-wrap; background: #f3f4f6; padding: 0.7rem; border-radius: 8px; font-size: 0.85rem; }
    .status { min-height: 1.2rem; color: #2563eb; font-size: 0.92rem; }
    .status.error { color: #dc2626; }
  </style>
</head>
<body>
  <h1>AI Product Discovery</h1>
  <p class="subtitle">Upload a feedback CSV. Get clusters, a draft PRD, and Jira tickets.</p>

  <div class="card">
    <label for="fileInput"><strong>Upload CSV</strong></label>
    <p class="muted" style="margin-top:0.25rem;">CSV must contain a <code>text</code> column (or legacy <code>feedback</code>). Optional columns: <code>feedback_id</code>, <code>source</code>.</p>
    <input id="fileInput" type="file" accept=".csv" />
    <button id="analyzeBtn">Analyze</button>
    <p id="status" class="status"></p>
    <div class="links">
      <a id="prdLink" class="disabled" target="_blank" rel="noopener">Download PRD</a>
      <a id="jiraLink" class="disabled" target="_blank" rel="noopener">Download Jira tickets</a>
    </div>
  </div>

  <div class="card">
    <h2 style="margin-top:0;">Top opportunities</h2>
    <div id="clusters"><p class="muted">No analysis yet.</p></div>
  </div>

  <div class="card">
    <h2 style="margin-top:0;">Recommended action</h2>
    <p id="action" class="muted">No analysis yet.</p>
  </div>

  <div class="card">
    <h2 style="margin-top:0;">Raw response</h2>
    <details>
      <summary class="muted">Show JSON</summary>
      <pre id="raw">No analysis yet.</pre>
    </details>
  </div>

  <script>
    const analyzeBtn = document.getElementById('analyzeBtn');
    const fileInput = document.getElementById('fileInput');
    const statusEl = document.getElementById('status');
    const clustersEl = document.getElementById('clusters');
    const actionEl = document.getElementById('action');
    const rawEl = document.getElementById('raw');
    const prdLink = document.getElementById('prdLink');
    const jiraLink = document.getElementById('jiraLink');

    function renderClusters(data) {
      const top = (data.top_opportunities || []).slice(0, 3);
      if (!top.length) {
        clustersEl.innerHTML = '<p class="muted">No clusters identified.</p>';
        return;
      }
      clustersEl.innerHTML = '';
      top.forEach((cluster, idx) => {
        const card = document.createElement('div');
        card.className = 'cluster';
        const theme = cluster.theme_label || `Cluster ${cluster.cluster_id}`;
        const freq = cluster.frequency ?? cluster.size ?? 0;
        const score = cluster.opportunity_score != null ? cluster.opportunity_score.toFixed(2) : '-';
        const example = cluster.example_signal || cluster.representative_feedback || '';
        card.innerHTML =
          `<h3>#${idx + 1}. ${theme}</h3>` +
          `<div class="meta">Frequency: ${freq} · Severity: ${cluster.severity ?? '-'} · Opportunity score: ${score}</div>` +
          (example ? `<div class="quote">"${example}"</div>` : '');
        clustersEl.appendChild(card);
      });

      if ((data.evidence || []).length) {
        const ev = document.createElement('div');
        ev.className = 'cluster';
        ev.innerHTML = '<h3>Supporting evidence</h3>' +
          (data.evidence || []).map(q => `<div class="quote">"${q}"</div>`).join('');
        clustersEl.appendChild(ev);
      }
    }

    analyzeBtn.addEventListener('click', async () => {
      const file = fileInput.files[0];
      if (!file) {
        statusEl.className = 'status error';
        statusEl.textContent = 'Please choose a CSV file first.';
        return;
      }

      const formData = new FormData();
      formData.append('file', file);

      analyzeBtn.disabled = true;
      statusEl.className = 'status';
      statusEl.textContent = 'Analyzing…';
      prdLink.classList.add('disabled');
      jiraLink.classList.add('disabled');
      prdLink.removeAttribute('href');
      jiraLink.removeAttribute('href');

      try {
        const response = await fetch('/analyze', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || 'Request failed');
        }
        statusEl.textContent = `Analysis complete. Run ID: ${data.run_id}`;
        actionEl.textContent = data.recommended_action || '';
        actionEl.classList.remove('muted');
        renderClusters(data);
        rawEl.textContent = JSON.stringify(data, null, 2);

        if (data.run_id) {
          prdLink.href = `/download/prd/${data.run_id}`;
          jiraLink.href = `/download/jira/${data.run_id}`;
          prdLink.classList.remove('disabled');
          jiraLink.classList.remove('disabled');
        }
      } catch (error) {
        statusEl.className = 'status error';
        statusEl.textContent = `Error: ${error.message}`;
      } finally {
        analyzeBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return HOME_HTML


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    if not (file.filename or "").endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    content = await file.read()
    run_id = uuid4().hex

    try:
        response = analyze_feedback_csv(content, run_id=run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to analyze CSV: {exc}") from exc

    _store_run(run_id, response.prd_text, response.jira_tickets_text)
    return response


@app.get("/download/prd/{run_id}")
def download_prd(run_id: str) -> PlainTextResponse:
    run = RUN_STORE.get(run_id)
    if not run or not run.get("prd"):
        raise HTTPException(status_code=404, detail="Unknown or expired run_id")
    return PlainTextResponse(
        content=run["prd"],
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="PRD.md"'},
    )


@app.get("/download/jira/{run_id}")
def download_jira(run_id: str) -> PlainTextResponse:
    run = RUN_STORE.get(run_id)
    if not run or not run.get("jira"):
        raise HTTPException(status_code=404, detail="Unknown or expired run_id")
    return PlainTextResponse(
        content=run["jira"],
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="jira_tickets.md"'},
    )
