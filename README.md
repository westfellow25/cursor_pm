# AI Product Discovery MVP

This MVP now runs entirely from FastAPI (no Node/npm required). The backend serves a minimal vanilla JS UI for CSV upload, analysis, and artifact downloads.

## API + UI endpoints

- `GET /` → HTML UI with:
  - CSV file upload
  - Analyze button
  - JSON results display
  - Download links for PRD/Jira markdown
- `POST /analyze` → runs analysis pipeline from uploaded CSV (`file` form field)
- `GET /download/prd` → download latest generated PRD markdown
- `GET /download/jira` → download latest generated Jira markdown
- `GET /health` → health check

## Quick start (Windows-friendly)

From the repository root:

```powershell
py -m pip install -r requirements.txt
py -m uvicorn backend.main:app --reload
```

Then open:

- <http://127.0.0.1:8000>

## CSV requirements

`POST /analyze` uses the existing analysis pipeline input format:

```csv
feedback_id,text,source
f001,"The dashboard is slow when loading monthly reports",web
```

## Notes

- `OPENAI_API_KEY` must be set in your environment for embedding/analysis steps.
- Download links become available after a successful `/analyze` run.
