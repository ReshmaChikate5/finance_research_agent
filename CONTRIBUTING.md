# Contributing

1. Create a branch: `git checkout -b feat/<name>`
2. Install hooks: `uv run pre-commit install`
3. Run tests: `uv sync --project agent_host --extra local`, then `uv run --project agent_host --extra local pytest`
4. Open a PR using the template.

markdown
## Branches
feat/<name>   fix/<name>   chore/<name>   docs/<name>
e.g. feat/sec-edgar-server

## Commits (Conventional Commits)
feat: add search_company_filings tool
fix: handle empty filings response
chore: bump ruff version
docs: document local setup
refactor: split planner node
test: add router unit tests

## PRs
One logical unit of work per PR (e.g. one MCP server per PR).
Squash-merge into main.