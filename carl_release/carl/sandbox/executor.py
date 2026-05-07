"""Subprocess Python sandbox with auto-display and injected search()."""
import ast, base64, json, os, re, subprocess, sys, tempfile, textwrap

TIMEOUT_SECONDS = 5


def _build_search_func(results: dict[str, str]) -> str:
    enc = base64.b64encode(json.dumps(results, ensure_ascii=True).encode()).decode()
    return textwrap.dedent(f"""\
        import json as _json, base64 as _b64
        _SR = _json.loads(_b64.b64decode("{enc}").decode())
        def search(query):
            return _SR.get(query, "No results found for: " + query)
    """)


def extract_search_query_strings(code: str) -> list[str]:
    out = []
    for a, b in re.findall(r'''search\(\s*(?:"([^"]+)"|'([^']+)')\s*\)''', code):
        q = a or b
        if q and q not in out:
            out.append(q)
    return out


def _auto_display(code: str) -> str:
    """Wrap last bare expression in print(), Jupyter-style."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    last = tree.body[-1]
    if (isinstance(last.value, ast.Call) and isinstance(last.value.func, ast.Name)
            and last.value.func.id == "print"):
        return code
    lines = code.split("\n")
    s, e = last.lineno - 1, last.end_lineno
    expr = "\n".join(lines[s:e]).strip()
    lines[s:e] = [f"print({expr})"]
    return "\n".join(lines)


def execute_code(code: str,
                 search_results: dict[str, str] | None = None,
                 timeout: int = TIMEOUT_SECONDS) -> str:
    if not code.strip():
        return ""
    code = _auto_display(code)
    script = ""
    if search_results:
        script += _build_search_func(search_results)
    script += "\n" + code
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(script)
            tmp_path = f.name
        r = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True,
                           timeout=timeout, cwd=tempfile.gettempdir())
        if r.returncode != 0:
            for line in reversed(r.stderr.strip().splitlines()):
                if line and not line.startswith(" ") and not line.startswith("Traceback"):
                    return f"ERROR: {line}"
            return "ERROR: Code execution failed"
        out = r.stdout.rstrip("\n")
        return out if out else "ERROR: Code produced no output"
    except subprocess.TimeoutExpired:
        return f"ERROR: Execution timed out ({timeout}s limit)"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except OSError: pass
