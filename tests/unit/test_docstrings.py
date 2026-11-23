"""Documentation standard (Step 20): every MCP tool has a complete docstring.

Reads the server files as text (no imports, no network), finds each function decorated with
@mcp.tool, and checks the docstring has: a one-line purpose, an Args section naming every
parameter, a Returns section, and an Example. Add a tool without these and CI fails.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SERVERS = sorted((Path(__file__).resolve().parents[2] / "mcp_servers").glob("*/server.py"))


def tools(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and any(
            "mcp.tool" in ast.unparse(d) for d in node.decorator_list
        ):
            yield node


CASES = [(p.parent.name, n) for p in SERVERS for n in tools(p)]


def test_all_four_servers_are_found():
    assert {p.parent.name for p in SERVERS} >= {
        "sec_edgar",
        "market_data",
        "portfolio",
        "notify_slack",
    }
    assert len(CASES) >= 9


@pytest.mark.parametrize("server,node", CASES, ids=[f"{s}.{n.name}" for s, n in CASES])
def test_tool_docstring_is_complete(server, node):
    doc = ast.get_docstring(node) or ""
    where = f"{server}.{node.name}"
    first = doc.strip().splitlines()[0] if doc.strip() else ""
    assert len(first) >= 15, f"{where}: start with a one-line purpose"
    for section in ("Args:", "Returns:", "Example"):
        assert section in doc, f"{where}: docstring has no '{section}' section"
    args_text = doc.split("Args:", 1)[1].split("Returns:", 1)[0]
    params = [a.arg for a in node.args.args + node.args.kwonlyargs if a.arg != "self"]
    for name in params:
        assert re.search(rf"\b{re.escape(name)}\b", args_text), (
            f"{where}: parameter '{name}' is not described under Args:"
        )
    example = doc.split("Example", 1)[1]
    assert node.name in example, f"{where}: the Example should show a call to {node.name}"
