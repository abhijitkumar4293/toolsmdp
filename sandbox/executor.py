"""Sandboxed code execution. Runs code in a subprocess with a timeout.

No import restrictions — the 5-second timeout is the safety boundary.
Jupyter-style auto-display: if the last statement is a bare expression,
it gets wrapped in print() so the model doesn't need explicit print().
"""

import ast
import json
import re
import subprocess
import sys
import textwrap
import tempfile
import os

TIMEOUT_SECONDS = 5

_SEARCH_PLACEHOLDER = textwrap.dedent("""\
    def search(query):
        return f"[Search results for: {query}] No search backend configured."
""")


def _build_search_func(search_results: dict[str, str]) -> str:
    """Build a search() function that returns pre-resolved results."""
    import base64
    results_json = json.dumps(search_results, ensure_ascii=True)
    encoded = base64.b64encode(results_json.encode()).decode()
    return textwrap.dedent(f"""\
        import json as _json, base64 as _b64
        _SEARCH_RESULTS = _json.loads(_b64.b64decode("{encoded}").decode())
        def search(query):
            return _SEARCH_RESULTS.get(query, "No results found for: " + query)
    """)


def extract_search_queries(code: str) -> list[str]:
    """Extract search() query strings from code. Handles search("...") and search('...')."""
    return re.findall(r'''search\(\s*(?:"([^"]+)"|'([^']+)')\s*\)''', code)


def extract_search_query_strings(code: str) -> list[str]:
    """Return deduplicated list of search query strings found in code."""
    queries = []
    for match in extract_search_queries(code):
        q = match[0] or match[1]
        if q and q not in queries:
            queries.append(q)
    return queries


def _auto_display(code: str) -> str:
    """Jupyter-style auto-display: wrap last bare expression in print()."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body:
        return code
    last = tree.body[-1]
    if not isinstance(last, ast.Expr):
        return code
    # Skip if already a print() call
    if (isinstance(last.value, ast.Call)
            and isinstance(last.value.func, ast.Name)
            and last.value.func.id == "print"):
        return code
    lines = code.split("\n")
    start = last.lineno - 1
    end = last.end_lineno
    expr_text = "\n".join(lines[start:end]).strip()
    lines[start:end] = [f"print({expr_text})"]
    return "\n".join(lines)


def execute_code(
    code: str,
    search_enabled: bool = False,
    search_results: dict[str, str] | None = None,
    timeout: int = TIMEOUT_SECONDS,
) -> str:
    """Execute Python code in a sandboxed subprocess. Returns stdout or 'ERROR: ...'.

    No import restrictions. The 5-second timeout is the safety net.
    If code errors or produces no output, returns a descriptive error.
    """
    if not code.strip():
        return ""

    code = _auto_display(code)

    script = ""
    if search_results:
        script += _build_search_func(search_results)
    elif search_enabled:
        script += _SEARCH_PLACEHOLDER
    script += "\n" + code

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(script)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
            cwd=tempfile.gettempdir(),
        )

        if result.returncode != 0:
            for line in reversed(result.stderr.strip().splitlines()):
                if line and not line.startswith(" ") and not line.startswith("Traceback"):
                    return f"ERROR: {line}"
            return "ERROR: Code execution failed"

        output = result.stdout.rstrip("\n")
        return output if output else "ERROR: Code produced no output"

    except subprocess.TimeoutExpired:
        return "ERROR: Execution timed out (5s limit)"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except (OSError, UnboundLocalError):
            pass
