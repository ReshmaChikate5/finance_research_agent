"""Scaffold the agentic-finance-agent repo structure.

Usage (from inside your git repo):
    python scaffold.py                 # create in current directory
    python scaffold.py --root my-dir   # create in ./my-dir
    python scaffold.py --dry-run       # preview only

Existing files are never overwritten.
"""

import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Files with real starter content
# ---------------------------------------------------------------------------
CONTENT = {
    ".gitignore": """\
.venv/
__pycache__/
*.pyc
.env
.aws/
.pytest_cache/
.ruff_cache/
.mypy_cache/
cdk.out/
node_modules/
chroma_db/
*.egg-info/
.DS_Store
""",
    ".env.example": """\
# --- LLM / AWS ---
AWS_PROFILE=myproject
AWS_REGION=ap-south-1

# --- Vector store: chroma | bedrock_kb ---
VECTOR_STORE=chroma
CHROMA_PATH=./chroma_db
BEDROCK_KB_ID=

# --- MCP servers ---
SEC_EDGAR_URL=http://localhost:8001
MARKET_DATA_URL=http://localhost:8002
PORTFOLIO_URL=http://localhost:8003
NOTIFY_SLACK_URL=http://localhost:8004

# --- Integrations ---
SLACK_WEBHOOK_URL=
MARKET_DATA_API_KEY=
SEC_USER_AGENT="Your Name your@email.com"
""",
    "docker-compose.yml": """\
services:
  agent_host:
    build: ./agent_host
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [sec_edgar, market_data, portfolio, notify_slack]

  sec_edgar:
    build: ./mcp_servers/sec_edgar
    ports: ["8001:8000"]
    env_file: .env

  market_data:
    build: ./mcp_servers/market_data
    ports: ["8002:8000"]
    env_file: .env

  portfolio:
    build: ./mcp_servers/portfolio
    ports: ["8003:8000"]
    env_file: .env

  notify_slack:
    build: ./mcp_servers/notify_slack
    ports: ["8004:8000"]
    env_file: .env
""",
    ".pre-commit-config.yaml": """\
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-yaml
""",
    "pyproject.toml": """\
[project]
name = "agentic-finance-agent"
version = "0.1.0"
description = "Agentic finance research assistant (LangGraph + MCP + AWS)"
requires-python = ">=3.11"

[dependency-groups]
dev = ["pytest", "ruff", "mypy", "pre-commit"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
""",
    "README.md": """\
# agentic-finance-agent

LangGraph-based finance research agent with MCP tool servers, local (Chroma)
and AWS (Bedrock KB) vector stores, and an AWS CDK deployment.

## Quick start

```bash
cp .env.example .env
uv sync
docker compose up --build
```
""",
    "CONTRIBUTING.md": """\
# Contributing

1. Create a branch: `git checkout -b feat/<name>`
2. Install hooks: `uv run pre-commit install`
3. Run tests: `uv run pytest`
4. Open a PR using the template.
""",
    ".github/pull_request_template.md": """\
## What does this PR do?

## How was it tested?

## Checklist
- [ ] Tests added/updated
- [ ] Docs updated
- [ ] No secrets committed
""",
    ".github/workflows/ci.yml": """\
name: CI
on:
  pull_request:
  push:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run pytest
""",
    ".github/workflows/deploy.yml": """\
name: Deploy
on:
  workflow_dispatch:
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      # TODO: configure AWS OIDC role, then run `cdk deploy --all`
""",
}


def dockerfile(port: int = 8000) -> str:
    return f"""\
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml ./
RUN uv sync --no-dev
COPY . .
EXPOSE {port}
CMD ["uv", "run", "python", "server.py"]
"""


def service_pyproject(name: str) -> str:
    return f"""\
[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["mcp"]
"""


# ---------------------------------------------------------------------------
# Structure: path -> short description (used for placeholder comments)
# ---------------------------------------------------------------------------
FILES = {
    # agent_host
    "agent_host/app/main.py": "FastAPI entrypoint",
    "agent_host/app/config.py": "Settings loaded from environment",
    "agent_host/app/graph/state.py": "LangGraph state schema",
    "agent_host/app/graph/nodes.py": "planner, synthesizer, human_approval nodes",
    "agent_host/app/graph/router.py": "conditional edge / tool router",
    "agent_host/app/graph/build_graph.py": "graph assembly",
    "agent_host/app/vectorstore/base.py": "VectorStore interface",
    "agent_host/app/vectorstore/chroma_store.py": "local Chroma implementation",
    "agent_host/app/vectorstore/bedrock_kb_store.py": "AWS Bedrock KB implementation",
    "agent_host/app/mcp_clients/.gitkeep": None,
    "agent_host/Dockerfile": dockerfile(8000).replace("server.py", "-m\", \"app.main"),
    "agent_host/pyproject.toml": service_pyproject("agent-host"),
    # mcp_servers
    "mcp_servers/sec_edgar/server.py": "SEC EDGAR MCP server",
    "mcp_servers/sec_edgar/Dockerfile": dockerfile(),
    "mcp_servers/sec_edgar/pyproject.toml": service_pyproject("sec-edgar"),
    "mcp_servers/market_data/server.py": "Market data MCP server",
    "mcp_servers/market_data/cache.py": "Caching layer for market data",
    "mcp_servers/market_data/Dockerfile": dockerfile(),
    "mcp_servers/market_data/pyproject.toml": service_pyproject("market-data"),
    "mcp_servers/portfolio/server.py": "Portfolio MCP server",
    "mcp_servers/portfolio/db.py": "Portfolio database access",
    "mcp_servers/portfolio/Dockerfile": dockerfile(),
    "mcp_servers/portfolio/pyproject.toml": service_pyproject("portfolio"),
    "mcp_servers/notify_slack/server.py": "Slack notification MCP server",
    "mcp_servers/notify_slack/Dockerfile": dockerfile(),
    "mcp_servers/notify_slack/pyproject.toml": service_pyproject("notify-slack"),
    # ingestion
    "ingestion/chunker.py": "Document chunking",
    "ingestion/embedder.py": "Embedding generation",
    "ingestion/load_chroma.py": "Load embeddings into local Chroma",
    # infra (AWS CDK)
    "infra/app.py": "CDK app entrypoint",
    "infra/stacks/network_stack.py": "VPC, subnets, security groups",
    "infra/stacks/secrets_stack.py": "Secrets Manager resources",
    "infra/stacks/data_stack.py": "Databases, buckets, knowledge base",
    "infra/stacks/compute_stack.py": "ECS/Fargate services",
    "infra/stacks/api_stack.py": "API Gateway / load balancer",
    "infra/constructs/mcp_server_construct.py": "Reusable MCP server construct",
    # tests / evals
    "tests/unit/.gitkeep": None,
    "tests/graph/.gitkeep": None,
    "evals/dataset.jsonl": "",
    "evals/run_evals.py": "Run evaluation dataset against the agent",
    # CI placeholders handled via CONTENT
}

# Folders that should be Python packages
PACKAGES = [
    "agent_host/app",
    "agent_host/app/graph",
    "agent_host/app/vectorstore",
    "agent_host/app/mcp_clients",
    "ingestion",
    "infra/stacks",
    "infra/constructs",
    "tests",
    "tests/unit",
    "tests/graph",
]


def render(path: str, value) -> str:
    """Return file content for a given path."""
    if path in CONTENT:
        return CONTENT[path]
    if value is None:
        return ""
    if path.endswith(".py"):
        return f'"""{value}."""\n'
    return value  # Dockerfiles / pyproject strings / empty files


def write(path: Path, content: str, dry_run: bool) -> None:
    if path.exists():
        print(f"  skip    {path}")
        return
    print(f"  create  {path}")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold agentic-finance-agent")
    parser.add_argument("--root", default=".", help="Target directory (default: current)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"Scaffolding into: {root}\n")

    # Merge: everything in FILES + everything in CONTENT
    all_paths = dict(FILES)
    for p in CONTENT:
        all_paths.setdefault(p, None)

    for rel, value in all_paths.items():
        write(root / rel, render(rel, value), args.dry_run)

    for pkg in PACKAGES:
        write(root / pkg / "__init__.py", "", args.dry_run)

    print("\nDone." if not args.dry_run else "\nDry run only, nothing written.")


if __name__ == "__main__":
    main()