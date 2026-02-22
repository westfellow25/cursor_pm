# AI Product Discovery MVP

Minimal MVP to ingest customer feedback, cluster similar problems using embeddings, and output opportunity summaries.

## 1) Proposed project structure

```text
.
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI entrypoint and API routes
│   │   ├── pipeline.py      # CSV loading + orchestration
│   │   ├── clustering.py    # KMeans clustering logic
│   │   ├── openai_client.py # OpenAI embeddings wrapper
│   │   └── schemas.py       # API response contracts
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Upload UI + results rendering
│   │   ├── main.jsx         # React bootstrap
│   │   └── styles.css
│   ├── index.html
│   └── package.json
└── data/
    └── sample_feedback.csv
```

## 2) Backend API

### Endpoints

- `GET /health` → health check.
- `POST /discover` → accepts CSV upload with required `feedback` column.

### Expected CSV

At least one column named `feedback`:

```csv
feedback
"Your customer feedback text here"
```

## 3) Clustering pipeline (MVP)

1. Read CSV and validate required columns.
2. Embed each feedback row via OpenAI (`text-embedding-3-small`).
3. Use a simple KMeans heuristic for cluster count (`sqrt(n)` bounded).
4. Build opportunity summaries with:
   - cluster size
   - short theme (first text snippet)
   - representative feedback example

## 4) Setup instructions

### Prerequisites

- Python 3.11+
- Node 18+
- `OPENAI_API_KEY` in environment

### Run backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your_key_here"
uvicorn app.main:app --reload --port 8000
```

### Run frontend

```bash
cd frontend
npm install
npm run dev
```

If backend runs on another host/port, set:

```bash
export VITE_API_URL="http://localhost:8000"
```

## Quick test

Use `data/sample_feedback.csv` in the UI upload form.


## 5) End-to-end demo script (simple + reliable)

This repository includes a single command demo that runs the full pipeline on the example CSV:

- loads `example_data/feedback.csv`
- runs ingestion
- runs clustering
- computes opportunity scores
- generates one feature recommendation
- prints everything clearly to the console

### Step-by-step

1. From the repository root, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Run the demo script:

```bash
python e2e_demo.py
```

Windows (PowerShell with Python launcher):

```powershell
py .\e2e_demo.py
```

### Expected output (high level)

You will see:

- number of feedback records loaded
- cluster IDs and record IDs per cluster
- top opportunity scores and summaries
- one recommended feature with supporting quotes
