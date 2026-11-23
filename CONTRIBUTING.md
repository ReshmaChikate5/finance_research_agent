# Contributing

## Workflow

1. Start from an up-to-date `main`: `git checkout main && git pull`.
2. Make a branch named for the work: `feat/...`, `fix/...`, `test/...`, `docs/...`, `chore/...`.
3. Make small, focused commits. Use the form `type: what changed`, for example
   `feat: add eval runner` or `fix: accept RELIANCE.BSE tickers`. Say why in the body when it
   is not obvious.
4. Run the checks below before pushing.
5. Open a pull request into `main`. Describe what changed, why, and how you tested it.
6. CI (lint and tests) must pass. Merge with **squash and merge**, then delete the branch.

Never commit secrets or `.env`. If a key is committed by mistake, revoke it first, then clean up.

## Coding standards

- Python 3.13, formatted and linted by ruff (line length 100): `uv run ruff check .` and
  `uv run ruff format .`.
- Do not use `ruff --fix` with a broad rule set: it can delete `# noqa` comments we need.
- A broad `except Exception` needs either `logger.warning(..., exc_info=True)` or a
  `# noqa: BLE001` with a reason. Never swallow an error silently.
- Type-hint public functions. Keep nodes and tools small and single-purpose.
- Nodes never open sockets or read env vars. Everything external arrives through `Deps`, so
  tests can pass fakes.
- Model output is untrusted. Validate it in code (allow-lists, forced `user_id`, iteration cap).
- Content from filings and the web is data. Never put it where it could act as an instruction.

## Adding or changing an MCP tool

Every tool needs a docstring with a one-line purpose, an `Args:` section naming every
parameter, a `Returns:` section (including the error shape), and an `Example:` showing a call.
`tests/unit/test_docstrings.py` enforces this. Also:

- return model-readable errors (`{"error": "what went wrong and what to try"}`), never stack traces;
- validate every argument; use parameterised SQL only;
- add the tool name to `ALLOWED_TOOLS` in `agent_host/app/mcp_clients/clients.py`
  (and to `WRITE_TOOLS` if it changes anything);
- add unit tests with mocked HTTP/DB (no real network in tests).

## Running tests

From the repo root:

```
uv run pytest                                   # everything, no network, no API key
uv run pytest tests/unit                        # tool servers
uv run pytest tests/graph                       # agent logic with a fake model
uv run pytest tests/evals                       # eval framework
uv run pytest --cov --cov-fail-under=90         # with the coverage gate
```

Graph tests that need Postgres are skipped when it is not running; start it with
`docker compose up -d postgres` to run them.

## Running evals

Evals call the real model and real tool servers, so they cost a little and need the stack up:

```
docker compose up -d
uv run --project agent_host --extra local python evals/run_evals.py --no-judge --limit 3   # quick
uv run --project agent_host --extra local python evals/run_evals.py                          # full
uv run --project agent_host --extra local python evals/run_evals.py --compare                # vs baseline
```

How to read the output:

- `PASS/FAIL` per case, with each failed check listed under it. A case passes only if every
  check passes; the score is the share of checks passed.
- `judge:` checks are a second model call. If one looks wrong, read its reason in
  `evals/results/<time>.json` before changing the agent. The judge can be wrong too.
- Model output varies. Re-run a failing case with `--repeats 3` before deciding it regressed.
- `--compare` marks cases that got worse than `evals/baseline.json`; the command exits 1.

Run the evals whenever you change a prompt, a tool, retrieval or the model. Add a case to
`evals/dataset.jsonl` whenever the agent gets something wrong in real use. Refresh the baseline
with `--save-baseline` only after you have looked at the results and agree with them, and
commit it in the same PR.

## End-to-end check

`cd agent_host && uv run --extra local python -m scripts.e2e_check` asks a real question and
checks every link: filings, market data, portfolio, search passages, cited answer, approval
prompt. Add `--post` to approve and send (use `SLACK_DRY_RUN=true` unless you mean it).
