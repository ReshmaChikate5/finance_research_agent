# Evals (Step 18)

Unit tests check that code does what we told it. Evals check that the *agent* gives good
answers. Run them before and after any change to prompts, tools, retrieval or the model.

## Run

From the repo root, Docker stack up (`docker compose up -d`), `ANTHROPIC_API_KEY` set:

    uv run --project agent_host --extra local python evals/run_evals.py --no-judge --limit 3
    uv run --project agent_host --extra local python evals/run_evals.py
    uv run --project agent_host --extra local python evals/run_evals.py --save-baseline
    uv run --project agent_host --extra local python evals/run_evals.py --compare

Other flags: `--ids a,b`, `--category rag`, `--repeats 3`, `--judge-model <id>`,
`--min-pass-rate 0.8` (exit code 1 if lower), `--timeout 240`.

Nothing is ever posted to Slack: the eval "human" answers no to every posting question and the
Slack client is replaced by one that raises.

## What is checked

Scripted checks (exact and free), from each case's `expect` block, plus these on every case:
run completed, nothing posted, within the iteration cap, disclaimer present, every `[S#]`
citation is listed under Sources.

Judge checks: each `rubric` line is a yes/no question answered by a second model call that sees
the question, the evidence the agent had, and the answer. Use a *different* model with
`--judge-model` when you can; a model tends to rate its own writing too kindly.

A case passes only if every counted check passes. Score = share of checks passed. A judge reply
that cannot be read is reported and not counted.

## Dataset (`dataset.jsonl`)

One JSON object per line. `id`, `category`, `question` are required; `user_id` defaults to 1.
`expect` keys: `tools_called`, `tools_called_any`, `tools_not_called` (patterns like
`portfolio.*`), `tool_args` (`{call, arg, equals}`), `portfolio_user_id`, `max_tool_calls`,
`min_distinct_tools`, `used_rag`, `has_citations`, `text_contains_all`, `text_contains_any`,
`text_not_contains`.

It assumes the Step 10 seed data (user 1 holds NVDA 50, AAPL 30, MSFT 20; user 2 holds AMD 100;
NVDA average cost 36.42) and only the NVDA 10-K in Chroma. Change the cases if your data differs.
Add a case whenever the agent gets something wrong in real use.

## Reading results

`evals/results/<time>.json` has every check, tools used and the answer text per run. The
terminal shows failed checks under each case. Model output varies, so a single failure is a
hint; compare `--repeats 3` runs before concluding something broke.
