# AI Product Discovery MVP

Upload a CSV of user feedback → get ranked opportunity clusters, a draft PRD, and a set of Jira tickets.

Runs entirely from FastAPI. No Node/npm, no OpenAI key required (the default pipeline uses a local hashing embedder).

## API + UI

- `GET /` — minimal vanilla JS UI: CSV upload, analysis result cards, PRD/Jira download links.
- `POST /analyze` — runs the analysis pipeline on the uploaded CSV (`file` form field). Returns JSON including a `run_id`.
- `GET /download/prd/{run_id}` — PRD markdown for a specific run.
- `GET /download/jira/{run_id}` — Jira tickets markdown for a specific run.
- `GET /samples` — list available demo datasets.
- `GET /samples/{name}` — return the CSV for a named demo dataset (`saas`, `ecommerce`, `fintech`).
- `GET /health` — health check.

The home page exposes "Try sample" buttons that load each bundled demo dataset and run it through `/analyze` without any upload.

## Quick start

```bash
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload
```

Then open <http://127.0.0.1:8000>.

## CSV format

Only the feedback text is required.

| column        | required | notes                                                                 |
|---------------|----------|-----------------------------------------------------------------------|
| `text`        | yes      | The raw feedback string. Accepts legacy column name `feedback` too.   |
| `feedback_id` | no       | Auto-generated as `f001`, `f002`, … if missing.                       |
| `source`      | no       | Defaults to `unknown`.                                                |

Example:

```csv
text,source
"The onboarding flow is confusing for first-time users",web
"I can't connect Slack and there is no useful error message",support
```

## Tests

```bash
pip install pytest httpx
python -m pytest
```

## Notes

- Per-run artifacts (PRD, Jira) are stored in memory keyed by `run_id`. The UI surfaces the `run_id` after each analysis so multiple users don't overwrite each other's downloads.
- Cluster labels and the proposed solution are derived from the actual data (TF-IDF over the cluster vs. the corpus) — no hardcoded themes.
