# Finance Research Agent

An AI research assistant for public-company analysis. Ask a question in plain English
("Give me a research brief on NVDA") and it gathers evidence from SEC filings, market data and
your own portfolio, writes a cited answer, and, only after a human approves, posts it to Slack.

## How it works

```
question -> planner (model picks ONE next step, code validates it)
              |-- SEC EDGAR tools   (filings, company financials)
              |-- market data tools (quotes, fundamentals, indicators)
              |-- portfolio tools   (holdings, cost basis; read-only)
              '-- filing search     (RAG over a local Chroma index)
         -> synthesizer (cited answer + disclaimer)
         -> human checkpoints (flagged draft, Slack approval, any write)
         -> Slack post (idempotent)
```

- **Agent host** (`agent_host/`): a LangGraph state machine. Progress is saved to Postgres after
  every step, so a crashed or paused run can be resumed.
- **Four MCP tool servers** (`mcp_servers/`): `sec_edgar`, `market_data`, `portfolio`,
  `notify_slack`. The model can never call the Slack server; only the approved-post step can.
- **Ingestion** (`ingestion/`): downloads filings, splits them by section, embeds them, and
  loads Chroma.
- **Evals** (`evals/`): a dataset and runner that score the agent's answers.

Safety rules the code enforces: the model may only use allow-listed tools; `user_id` is set by the
system, never by the model; the portfolio database role is read-only; filings are treated as data,
never as instructions; the loop has a hard iteration cap.

## Requirements

- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- Docker Desktop (Postgres and the four tool servers run in containers)
- API keys: Anthropic (the model), Finnhub (quotes), optionally Alpha Vantage (indicators,
  25 calls/day on the free tier), and a Slack incoming-webhook URL (or use dry-run mode)

## Setup

1. Copy `.env.example` to `.env` and fill in the values (never commit `.env`). The ones you
   need:

   | Variable | Used for |
   |---|---|
   | `ANTHROPIC_API_KEY` | the model |
   | `FINNHUB_API_KEY` / `ALPHA_VANTAGE_API_KEY` | market data |
   | `SEC_USER_AGENT` | required by SEC, e.g. `Your Name you@example.com` |
   | `SLACK_WEBHOOK_URL`, `SLACK_CHANNEL_NAME` | posting reports |
   | `SLACK_DRY_RUN=true` | skip the real post while testing |
   | `CHROMA_PATH`, `CHROMA_COLLECTION` | where the filing index lives |
   | `CHECKPOINT_DATABASE_URL`, `PORTFOLIO_DATABASE_URL` | Postgres connections |

   The full list with defaults is in `.env.example` and `agent_host/app/config.py`.

2. Start the stack and check it is healthy:

   ```
   docker compose up -d --build
   docker compose ps
   ```

3. Load some filings into the search index (run from the repo root):

   ```
   uv run --project agent_host --extra local python -m ingestion.load_chroma --tickers NVDA --forms 10-K --limit 1
   ```

## Run it

From `agent_host/`:

```
uv run --extra local python -m scripts.run_agent "Give me a research brief on NVDA"
uv run --extra local python -m scripts.run_agent "Brief on NVDA" --no-notify   # never posts
uv run --extra local python -m scripts.run_agent "Brief on NVDA" --pause       # answer later
uv run --extra local python -m scripts.run_agent --resume THREAD_ID --answer yes
uv run --extra local python -m scripts.e2e_check                               # checks every link
```

The agent stops and asks you before it posts to Slack. Answer `n` and nothing is sent.

## Test

From the repo root:

```
uv run ruff check .
uv run pytest                                  # unit + graph + eval-framework tests, no network
uv run --project agent_host --extra local python evals/run_evals.py --no-judge --limit 3
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow, and
[evals/README.md](evals/README.md) for how scoring works.

## Repository layout

```
agent_host/        LangGraph agent, MCP clients, vector store, scripts
mcp_servers/       sec_edgar, market_data, portfolio, notify_slack (one server.py each)
ingestion/         filings -> chunks -> embeddings -> Chroma
sql/               database schema, roles and seed data
evals/             dataset.jsonl, scoring, runner, baseline.json
tests/             unit/ (tool servers), graph/ (agent logic), evals/
```

## Status

Local development is complete. AWS deployment (ECS Fargate, CDK, API Gateway) is the next part;
this README gains the deployed URL and an architecture diagram when that ships.

_Reports are for information only and are not financial advice._
