# Pulse — AI-powered Product Intelligence Platform

> Turn customer feedback into product strategy. Automatically.

Pulse is a multi-tenant SaaS platform that ingests customer feedback from every source (CSV, Intercom, Zendesk, Slack, REST API, App Store, G2) and turns it into a continuous product intelligence stream: clustered themes, temporal trends, AI-generated insights, and ready-to-ship artifacts (PRDs, Jira tickets, executive summaries).

## The Moat

1. **Data flywheel** — our proprietary feedback taxonomy improves with every customer processed.
2. **Network effects** — anonymised cross-company benchmarks: *"Companies in your vertical see 3× more onboarding complaints."*
3. **Temporal lock-in** — months of historical trend data that can't be reconstructed by competitors.
4. **Integration depth** — deep, bidirectional connectors create high switching costs.
5. **Domain AI** — models trained specifically on product-feedback language, not generic NLP.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  React SPA (Tailwind + Recharts)          │
│    Dashboard · Feedback · Insights · Trends · Artifacts   │
└──────────────────────┬──────────────────────────────────┘
                       │  REST + JWT
┌──────────────────────┴──────────────────────────────────┐
│                   FastAPI (pulse.main)                    │
├───────────────┬───────────────┬──────────────────────────┤
│ Ingestion     │ Intelligence   │ Workflow                 │
│ - CSV         │ - Embeddings   │ - PRD / Jira generation  │
│ - Intercom    │ - Clustering   │ - Executive summary      │
│ - Slack       │ - Sentiment    │ - Impact tracking        │
│ - REST API    │ - Anomalies    │                          │
│ (+ registry)  │ - Trends       │                          │
└──────┬────────┴────────┬──────┴──────────────────────────┘
       │                 │
┌──────┴────────┐ ┌──────┴──────────────────────────────────┐
│ SQLAlchemy    │ │ Multi-tenant schema                      │
│ (SQLite dev,  │ │ Orgs · Users · Sources · Feedback       │
│  Postgres prod)│ │ Clusters · Insights · Artifacts · Trends │
└───────────────┘ └──────────────────────────────────────────┘
```

## Project Layout

```
cursor_pm/
├── pulse/                   # Core platform package
│   ├── config.py            # Pydantic settings
│   ├── database.py          # SQLAlchemy engine / session
│   ├── models.py            # All domain models
│   ├── main.py              # FastAPI app entry
│   ├── ml/                  # Embeddings · sentiment · clustering · anomaly
│   ├── services/            # Ingestion · intelligence · trends · insights · artifacts
│   ├── connectors/          # CSV · Intercom · Slack · API (pluggable)
│   └── api/                 # Routes · schemas · JWT deps
├── frontend/                # React SPA
│   └── src/
│       ├── api/client.js    # Typed API client
│       ├── components/      # Layout, shared UI
│       └── pages/           # Dashboard, Feedback, Insights, Trends, Integrations, Artifacts
├── scripts/
│   └── seed.py              # Generate 6 months of realistic demo data
├── tests/                   # Unit tests
├── pyproject.toml
├── .env.example
└── README.md
```

## Quick Start

### 🚀 Option A: GitHub Codespaces (easiest — no local setup)

1. Open this repo on GitHub
2. Click **Code → Codespaces → Create codespace on main**
3. Wait ~2 minutes for auto-setup (deps install, seed data, frontend build)
4. Run: `uvicorn pulse.main:app --host 0.0.0.0 --port 8000`
5. When VS Code prompts to open the forwarded port 8000, click **Open in Browser**
6. Log in with `demo@acme.com` / `demo1234`

To enable AI features: edit `.env` and add `ANTHROPIC_API_KEY=sk-ant-...`

### 🐳 Option B: Docker (one command)

```bash
docker-compose up
```

Then open http://localhost:8000 — login with `demo@acme.com` / `demo1234`.

### 💻 Option C: Local install

#### 1. Backend

```bash
python -m pip install -e .
cp .env.example .env
# Edit .env: set SECRET_KEY and either ANTHROPIC_API_KEY (recommended)
#           or OPENAI_API_KEY for LLM features
python -m scripts.seed         # populate demo data (2500 items, 6 months)
uvicorn pulse.main:app --reload
```

### LLM provider selection

Pulse supports both **Anthropic Claude** and **OpenAI GPT**. Claude is preferred
for product/business analysis (longer context, better reasoning).

| Env var set | Active LLM |
|---|---|
| `ANTHROPIC_API_KEY` | Claude (recommended) |
| `OPENAI_API_KEY` only | GPT fallback |
| Neither | Heuristic fallback (still works, just less rich) |

The active provider is visible in the sidebar and at `GET /api/v1/system/status`.

Backend runs at **http://localhost:8000** — OpenAPI docs at `/docs`.

#### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Dev server runs at **http://localhost:5173** with API proxy to backend.

#### 3. Build for production

```bash
cd frontend && npm run build
# The FastAPI app will serve frontend/dist automatically.
uvicorn pulse.main:app --host 0.0.0.0 --port 8000
```

## Demo credentials (after running seed)

| Email | Password |
|-------|----------|
| `demo@acme.com` | `demo1234` |

Organisation: *Acme Corp* (SaaS, 51-200) with **2500 feedback items** across 5 sources over 6 months, drawn from **252 unique feedback templates** spanning 10 categories.

## Key API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/auth/signup` | Create org + admin user |
| POST | `/api/v1/auth/login` | Obtain JWT |
| GET | `/api/v1/dashboard` | Dashboard stats, trends, insights |
| GET | `/api/v1/feedback` | List / search / filter feedback |
| POST | `/api/v1/feedback` | Push feedback via API |
| POST | `/api/v1/feedback/upload` | Upload CSV |
| POST | `/api/v1/analysis/run` | Trigger clustering + insights + artifacts |
| GET | `/api/v1/analysis/latest` | Latest analysis with clusters |
| GET | `/api/v1/insights` | List AI-generated insights |
| GET | `/api/v1/trends` | Time-series trend data |
| GET | `/api/v1/artifacts` | List PRDs, Jira tickets, summaries |
| GET | `/api/v1/integrations/available` | Available connectors |
| POST | `/api/v1/integrations` | Connect a new source |

## How It Works

1. **Ingest** — feedback flows in via CSV, integrations, or API. Each item is auto-enriched with sentiment, urgency, category classification, and a 256-dim embedding.
2. **Cluster** — K-Means with silhouette-optimised k-selection groups semantically similar feedback into themes.
3. **Score** — each cluster gets an opportunity score combining frequency, severity, sentiment, and segment prevalence.
4. **Analyse trends** — weekly snapshots feed the temporal intelligence engine; anomalies surface as insights (volume spikes, sentiment shifts, emerging topics, segment divergences).
5. **Generate artifacts** — every analysis run auto-produces a PRD, 5-ticket Jira breakdown, and executive summary for the top opportunity.

## Next Steps (for scaling)

- Switch SQLite → Postgres + pgvector (embeddings stored natively)
- Add Celery + Redis for async analysis jobs
- Add webhook receiver for real-time ingestion from Intercom/Slack
- Ship cross-company benchmarks service (anonymised aggregations)
- Fine-tune a domain-specific sentiment model on proprietary feedback corpus
- Add bidirectional Jira/Linear sync with impact tracking
