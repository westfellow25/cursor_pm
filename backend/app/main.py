from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse

from .pipeline import analyze_feedback_csv
from .schemas import AnalyzeResponse

app = FastAPI(title="AI Product Discovery MVP", version="0.3.0")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = PROJECT_ROOT / "example_data"
SAMPLES: list[dict[str, str]] = [
    {
        "name": "saas",
        "filename": "sample_saas.csv",
        "label": "SaaS onboarding & integrations",
        "description": "14 support tickets: onboarding drop-off, Slack integration failures, CSV export gaps, billing clarity.",
    },
    {
        "name": "ecommerce",
        "filename": "sample_ecommerce.csv",
        "label": "E-commerce checkout & search",
        "description": "14 complaints: checkout payment errors, shipping transparency, product search quality, returns UX.",
    },
    {
        "name": "fintech",
        "filename": "sample_fintech.csv",
        "label": "Fintech login & transfers",
        "description": "14 app reviews: 2FA login pain, transfer limits/delays, statement export, KYC rejections.",
    },
]
SAMPLES_BY_NAME = {entry["name"]: entry for entry in SAMPLES}

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
    .cluster ul.items { margin: 0.4rem 0 0 1rem; padding: 0; color: #374151; }
    .cluster ul.items li { margin: 0.2rem 0; }
    details { margin-top: 0.5rem; }
    details pre { white-space: pre-wrap; background: #f3f4f6; padding: 0.7rem; border-radius: 8px; font-size: 0.85rem; }
    .status { min-height: 1.2rem; color: #2563eb; font-size: 0.92rem; }
    .status.error { color: #dc2626; }
    .dropzone { margin-top: 0.4rem; padding: 1.1rem; border: 2px dashed #c7d2fe; border-radius: 10px; text-align: center; color: #6b7280; cursor: pointer; background: #fafbff; transition: background 0.15s, border-color 0.15s; }
    .dropzone.dragging { background: #eef2ff; border-color: #6366f1; color: #1f2430; }
    .dropzone.has-file { border-style: solid; color: #1f2430; background: #f0fdf4; border-color: #86efac; }
    .preview { margin-top: 0.5rem; }
    .preview pre { white-space: pre-wrap; background: #f8fafc; border: 1px solid #e4e7eb; padding: 0.8rem; border-radius: 8px; font-size: 0.82rem; max-height: 340px; overflow: auto; }
  </style>
</head>
<body>
  <h1>AI Product Discovery</h1>
  <p class="subtitle">Upload a feedback CSV. Get clusters, a draft PRD, and Jira tickets.</p>

  <div class="card">
    <label for="fileInput"><strong>Upload CSV</strong></label>
    <p class="muted" style="margin-top:0.25rem;">CSV must contain a <code>text</code> column (or legacy <code>feedback</code>). Optional columns: <code>feedback_id</code>, <code>source</code>.</p>
    <div id="dropZone" class="dropzone">
      <span id="dropHint">Drop a CSV here or click to choose</span>
      <input id="fileInput" type="file" accept=".csv" hidden />
    </div>
    <div style="margin-top: 0.6rem; display: flex; gap: 0.5rem; align-items: center;">
      <button id="analyzeBtn">Analyze</button>
      <span id="fileName" class="muted" style="font-size: 0.9rem;"></span>
    </div>
    <p id="status" class="status"></p>
    <div style="margin-top:0.6rem;">
      <strong>Or try a sample dataset:</strong>
      <div id="samples" style="margin-top:0.4rem; display: flex; flex-wrap: wrap; gap: 0.4rem;"></div>
    </div>
    <div class="links" style="margin-top:0.9rem; display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap;">
      <a id="prdLink" class="disabled" target="_blank" rel="noopener">Download PRD</a>
      <button id="copyPrdBtn" type="button" disabled>Copy PRD</button>
      <a id="jiraLink" class="disabled" target="_blank" rel="noopener">Download Jira tickets</a>
      <button id="copyJiraBtn" type="button" disabled>Copy Jira</button>
      <span id="copyStatus" class="muted" style="font-size: 0.85rem;"></span>
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
    <h2 style="margin-top:0;">Artifacts</h2>
    <p class="muted" style="margin-top:0;">Generated PRD and Jira tickets from the latest analysis.</p>
    <details id="prdPreview" class="preview">
      <summary>PRD.md</summary>
      <pre id="prdPre">Run an analysis to see the PRD.</pre>
    </details>
    <details id="jiraPreview" class="preview">
      <summary>jira_tickets.md</summary>
      <pre id="jiraPre">Run an analysis to see the Jira tickets.</pre>
    </details>
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
    const dropZone = document.getElementById('dropZone');
    const dropHint = document.getElementById('dropHint');
    const fileNameEl = document.getElementById('fileName');
    const statusEl = document.getElementById('status');
    const clustersEl = document.getElementById('clusters');
    const actionEl = document.getElementById('action');
    const rawEl = document.getElementById('raw');
    const prdPre = document.getElementById('prdPre');
    const jiraPre = document.getElementById('jiraPre');
    const prdLink = document.getElementById('prdLink');
    const jiraLink = document.getElementById('jiraLink');
    const copyPrdBtn = document.getElementById('copyPrdBtn');
    const copyJiraBtn = document.getElementById('copyJiraBtn');
    const copyStatus = document.getElementById('copyStatus');
    let lastPrdText = '';
    let lastJiraText = '';

    function setSelectedFile(file) {
      if (!file) {
        fileNameEl.textContent = '';
        dropZone.classList.remove('has-file');
        dropHint.textContent = 'Drop a CSV here or click to choose';
        return;
      }
      fileNameEl.textContent = file.name + ' · ' + Math.round(file.size / 1024) + ' KB';
      dropZone.classList.add('has-file');
      dropHint.textContent = 'Ready: ' + file.name + ' (click to replace)';
    }

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
    dropZone.tabIndex = 0;
    fileInput.addEventListener('change', () => setSelectedFile(fileInput.files[0]));
    ['dragenter', 'dragover'].forEach(evt => dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragging');
    }));
    ['dragleave', 'drop'].forEach(evt => dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragging');
    }));
    dropZone.addEventListener('drop', (e) => {
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (!file) return;
      if (!file.name.toLowerCase().endsWith('.csv')) {
        statusEl.className = 'status error';
        statusEl.textContent = 'Please drop a .csv file.';
        return;
      }
      const dt = new DataTransfer();
      dt.items.add(file);
      fileInput.files = dt.files;
      setSelectedFile(file);
    });

    async function copyText(text, label) {
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
        copyStatus.textContent = `${label} copied to clipboard.`;
      } catch (err) {
        copyStatus.textContent = `Could not copy ${label}: ${err.message}`;
      }
      setTimeout(() => { copyStatus.textContent = ''; }, 2500);
    }

    copyPrdBtn.addEventListener('click', () => copyText(lastPrdText, 'PRD'));
    copyJiraBtn.addEventListener('click', () => copyText(lastJiraText, 'Jira tickets'));

    function escapeHtml(str) {
      return String(str)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }

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
        const theme = escapeHtml(cluster.theme_label || `Cluster ${cluster.cluster_id}`);
        const freq = cluster.frequency ?? cluster.size ?? 0;
        const score = cluster.opportunity_score != null ? cluster.opportunity_score.toFixed(2) : '-';
        const example = cluster.example_signal || cluster.representative_feedback || '';
        const texts = cluster.texts || [];
        const ids = cluster.ids || [];

        let membership = '';
        if (texts.length > 1) {
          const rows = texts.map((quote, i) => {
            const label = ids[i] ? `<span class="muted">[${escapeHtml(ids[i])}]</span> ` : '';
            return `<li>${label}${escapeHtml(quote)}</li>`;
          }).join('');
          membership = `<details><summary class="muted">Show all ${texts.length} items</summary><ul class="items">${rows}</ul></details>`;
        }

        card.innerHTML =
          `<h3>#${idx + 1}. ${theme}</h3>` +
          `<div class="meta">Frequency: ${freq} · Severity: ${cluster.severity ?? '-'} · Opportunity score: ${score}</div>` +
          (example ? `<div class="quote">"${escapeHtml(example)}"</div>` : '') +
          membership;
        clustersEl.appendChild(card);
      });

      if ((data.evidence || []).length) {
        const ev = document.createElement('div');
        ev.className = 'cluster';
        ev.innerHTML = '<h3>Supporting evidence (top cluster)</h3>' +
          (data.evidence || []).map(q => `<div class="quote">"${escapeHtml(q)}"</div>`).join('');
        clustersEl.appendChild(ev);
      }
    }

    async function runAnalysis(blob, filename, noticeLabel) {
      const formData = new FormData();
      formData.append('file', blob, filename);

      analyzeBtn.disabled = true;
      statusEl.className = 'status';
      statusEl.textContent = noticeLabel ? `Analyzing ${noticeLabel}…` : 'Analyzing…';
      prdLink.classList.add('disabled');
      jiraLink.classList.add('disabled');
      prdLink.removeAttribute('href');
      jiraLink.removeAttribute('href');
      copyPrdBtn.disabled = true;
      copyJiraBtn.disabled = true;
      lastPrdText = '';
      lastJiraText = '';

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
        lastPrdText = data.prd_text || '';
        lastJiraText = data.jira_tickets_text || '';
        copyPrdBtn.disabled = !lastPrdText;
        copyJiraBtn.disabled = !lastJiraText;
        prdPre.textContent = lastPrdText || 'No PRD generated.';
        jiraPre.textContent = lastJiraText || 'No Jira tickets generated.';
      } catch (error) {
        statusEl.className = 'status error';
        statusEl.textContent = `Error: ${error.message}`;
      } finally {
        analyzeBtn.disabled = false;
      }
    }

    analyzeBtn.addEventListener('click', () => {
      const file = fileInput.files[0];
      if (!file) {
        statusEl.className = 'status error';
        statusEl.textContent = 'Please choose a CSV file first.';
        return;
      }
      runAnalysis(file, file.name, '');
    });

    async function loadSamples() {
      const samplesEl = document.getElementById('samples');
      try {
        const response = await fetch('/samples');
        const data = await response.json();
        (data.samples || []).forEach((sample) => {
          const btn = document.createElement('button');
          btn.style.background = '#eef2ff';
          btn.style.color = '#1f2430';
          btn.style.border = '1px solid #c7d2fe';
          btn.textContent = `Try: ${sample.label}`;
          btn.title = sample.description;
          btn.addEventListener('click', async () => {
            const csvResp = await fetch(`/samples/${sample.name}`);
            if (!csvResp.ok) {
              statusEl.className = 'status error';
              statusEl.textContent = `Could not load sample ${sample.name}`;
              return;
            }
            const blob = await csvResp.blob();
            runAnalysis(blob, sample.filename || `${sample.name}.csv`, sample.label);
          });
          samplesEl.appendChild(btn);
        });
      } catch (error) {
        samplesEl.textContent = 'Samples unavailable.';
      }
    }
    loadSamples();
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


@app.get("/samples")
def list_samples() -> dict[str, list[dict[str, str]]]:
    return {
        "samples": [
            {"name": s["name"], "label": s["label"], "description": s["description"], "filename": s["filename"]}
            for s in SAMPLES
        ]
    }


@app.get("/samples/{name}")
def get_sample(name: str) -> PlainTextResponse:
    sample = SAMPLES_BY_NAME.get(name)
    if not sample:
        raise HTTPException(status_code=404, detail=f"Unknown sample: {name}")
    path = SAMPLES_DIR / sample["filename"]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Sample file missing on server: {sample['filename']}")
    return PlainTextResponse(
        content=path.read_text(encoding="utf-8"),
        media_type="text/csv",
        headers={"Content-Disposition": f'inline; filename="{sample["filename"]}"'},
    )


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
