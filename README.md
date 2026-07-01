# SHL Conversational Assessment Recommender

A stateless conversational API that recommends SHL assessments to hiring
managers based on their natural-language requirements.

---

## Project layout

```
shl-recommender/
├── app/
│   ├── __init__.py       # package marker
│   ├── main.py           # FastAPI app — GET /, /health, POST /chat
│   ├── schemas.py        # Pydantic v2 models (API contract)
│   ├── catalog.py        # loads and queries data/catalog.json
│   ├── retrieval.py      # BM25 search + ranking
│   ├── agent.py          # conversation controller (clarify/retrieve/compare/refuse)
│   └── llm.py            # LLM provider wrapper (Groq → Google → fallback)
├── data/
│   └── catalog.json      # ✅ committed — ships with deployed code (no runtime scrape)
├── eval/
│   ├── traces/           # 5 reference conversation traces (*.json)
│   └── replay_harness.py # offline Recall@10 evaluation script
├── render.yaml           # Render.com deployment spec
├── requirements.txt
└── README.md
```

---

## Quick start (local)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set at least one LLM API key (see "Environment variables" below)
#    Copy the example and fill in your keys:
copy .env.example .env

# 4. Run the dev server
uvicorn app.main:app --reload --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API docs.

---

## Environment variables

The app reads credentials from environment variables (or a local `.env` file via
`python-dotenv`). **Never commit `.env` — it is in `.gitignore`.**

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ recommended | Groq Cloud key — fastest provider, used first if present |
| `GOOGLE_API_KEY` | optional | Google Gemini key — fallback if Groq key is absent |

At least one of `GROQ_API_KEY` or `GOOGLE_API_KEY` must be set, or the app will
raise a `RuntimeError` on startup.

### Local `.env` file (never commit this)

```dotenv
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...
```

---

## Deploying to Render

The repo ships with a `render.yaml` that Render auto-detects on import.

### Steps

1. Push this repo to GitHub (or GitLab / Bitbucket).
2. In the [Render dashboard](https://dashboard.render.com/) → **New → Web Service**.
3. Connect your repo — Render detects `render.yaml` automatically.
4. On the **Environment** tab, add the two secret variables:

   | Key | Where to get it |
   |---|---|
   | `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) |
   | `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com/) → Get API key |

5. Click **Deploy**. Render will:
   - Run `pip install -r requirements.txt`
   - Start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Ping `GET /health` to confirm liveness before routing traffic

> **Free-tier note**: Render free web services spin down after 15 minutes of
> inactivity and take ~30 s to cold-start. The first request after a cold start
> may hit the agent's 25 s timeout; subsequent requests are fast.

### What is committed vs. excluded

| File | Committed? | Reason |
|---|---|---|
| `data/catalog.json` | ✅ Yes | Must ship with the image — no runtime scraper on Render |
| `.env` | ❌ No | Contains secrets — set vars in Render dashboard instead |
| `.venv/` | ❌ No | Render installs deps fresh from `requirements.txt` |
| `*.html` | ❌ No | Scraper artifacts — not needed in production |

---

## API contract

### `GET /`
Friendly landing response — confirms the service is up:
```json
{"message": "SHL Recommender API — see /docs", "docs": "/docs", "health": "/health"}
```

### `GET /health`
Returns `{"status": "ok"}` with HTTP 200. Used as Render's health-check endpoint.

### `POST /chat`

**Request**
```json
{
  "messages": [
    {"role": "user", "content": "I need a test for a mid-level software engineer."}
  ]
}
```

**Response**
```json
{
  "reply": "...",
  "recommendations": [
    {
      "name": "Verify Numerical Reasoning",
      "url": "https://www.shl.com/...",
      "test_type": "Ability & Aptitude"
    }
  ],
  "end_of_conversation": false
}
```

---

## Architecture overview

```
POST /chat
    │
    ▼
agent.run(request)
    │
    ├─ classify intent ──► llm.chat_completion()
    │
    ├─ CLARIFY  ──► llm.chat_completion()  ──► ChatResponse (no recs)
    ├─ RETRIEVE ──► retrieval.search()     ──► ChatResponse (with recs)
    ├─ COMPARE  ──► catalog.get_by_id()    ──► ChatResponse (comparison)
    └─ REFUSE   ──────────────────────────►  ChatResponse (no recs)
```

### Hard constraints

| Constraint | Value |
|---|---|
| Max conversation turns | 8 |
| Max response latency | 30 s (25 s agent budget + 5 s overhead) |
| Grounding | Catalog-only — no hallucinated assessments |

---

## Evaluation

```bash
# Against a local server:
python eval/replay_harness.py --url http://localhost:8000 --traces eval/traces

# Against the deployed Render URL:
python eval/replay_harness.py --url https://shl-recommender.onrender.com --traces eval/traces
```

Metrics reported per trace and in aggregate:
- **Recall@10** — fraction of gold assessments in the agent's top-10
- **Mean Recall@10** — arithmetic mean across all traces
- **Quality flags** — `EARLY_REC`, `BAD_URL`, `BAD_SCHEMA`
