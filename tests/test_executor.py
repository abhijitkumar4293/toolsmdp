from sandbox.executor import execute_code, _auto_display


class TestExecuteCode:

    def test_simple_arithmetic(self):
        assert execute_code("print(2 + 2)") == "4"

    def test_string_operation(self):
        assert execute_code('print("hello"[::-1])') == "olleh"

    def test_multiline_output(self):
        code = "for i in range(3):\n    print(i)"
        assert execute_code(code) == "0\n1\n2"

    def test_name_error(self):
        result = execute_code("pritn(42)")
        assert result.startswith("ERROR:")
        assert "NameError" in result

    def test_syntax_error(self):
        result = execute_code("def f(:")
        assert result.startswith("ERROR:")

    def test_zero_division(self):
        result = execute_code("print(1/0)")
        assert result.startswith("ERROR:")
        assert "ZeroDivision" in result

    def test_timeout(self):
        result = execute_code("while True: pass", timeout=1)
        assert result.startswith("ERROR:")
        assert "timed out" in result.lower()

    # All standard library imports are allowed
    def test_import_math(self):
        result = execute_code("import math\nprint(round(math.pi, 2))")
        assert result == "3.14"

    def test_import_collections(self):
        code = "from collections import Counter\nprint(Counter('aab'))"
        result = execute_code(code)
        assert "a" in result and "2" in result

    def test_import_json(self):
        code = 'import json\nprint(json.dumps({"a": 1}))'
        result = execute_code(code)
        assert '"a": 1' in result

    def test_import_os(self):
        result = execute_code("import os\nprint(os.getcwd())")
        assert not result.startswith("ERROR:")

    def test_import_sys(self):
        result = execute_code("import sys\nprint(sys.version[:5])")
        assert not result.startswith("ERROR:")

    def test_import_re(self):
        result = execute_code("import re\nprint(re.findall(r'\\d+', 'abc123def456'))")
        assert "123" in result

    def test_import_datetime(self):
        result = execute_code("from datetime import date\nprint(date.today().year)")
        assert not result.startswith("ERROR:")

    def test_empty_code(self):
        assert execute_code("") == ""
        assert execute_code("   ") == ""

    def test_no_output_returns_error(self):
        result = execute_code("x = 42")
        assert result.startswith("ERROR:")
        assert "no output" in result.lower()

    def test_search_disabled_by_default(self):
        result = execute_code("search('test')")
        assert result.startswith("ERROR:")

    def test_search_enabled_placeholder(self):
        result = execute_code("print(search('France GDP'))", search_enabled=True)
        assert "France GDP" in result

    def test_search_preresolved(self):
        result = execute_code(
            "print(search('capital of France'))",
            search_enabled=True,
            search_results={"capital of France": "Paris is the capital of France."},
        )
        assert "Paris" in result

    def test_large_output(self):
        code = "for i in range(1000):\n    print(i)"
        result = execute_code(code)
        assert "999" in result

    def test_float_precision(self):
        result = execute_code("print(0.15 * 3.05e12)")
        assert "457500000000" in result


class TestAutoDisplay:
    """Jupyter-style auto-display: bare last expressions get wrapped in print()."""

    def test_bare_variable(self):
        assert execute_code("x = 42\nx") == "42"

    def test_bare_arithmetic(self):
        assert execute_code("12 * (50 / 60)") == "10.0"

    def test_multi_line_bare_expr(self):
        assert execute_code("x = 48\ny = x / 2\nx + y") == "72.0"

    def test_assignment_not_displayed(self):
        """Last statement is assignment — no auto-display, still 'no output'."""
        result = execute_code("x = 42")
        assert "no output" in result.lower()

    def test_print_not_double_wrapped(self):
        """Already has print() — don't wrap again."""
        assert execute_code("print(42)") == "42"

    def test_search_bare_call(self):
        """search() without print() should auto-display."""
        result = execute_code(
            "search('capital of France')",
            search_enabled=True,
            search_results={"capital of France": "Paris is the capital."},
        )
        assert "Paris" in result

    def test_tuple_expression(self):
        assert execute_code("x = 1\ny = 2\n(x, y)") == "(1, 2)"

    def test_string_expression(self):
        assert execute_code("'hello' + ' world'") == "hello world"

    def test_syntax_error_passthrough(self):
        """Broken code should pass through unchanged (and error at execution)."""
        result = execute_code("def f(:\nx")
        assert result.startswith("ERROR:")

    def test_for_loop_not_touched(self):
        """Last statement is a for loop — not an Expr, no auto-display."""
        result = execute_code("for i in range(3):\n    pass")
        assert "no output" in result.lower()

    def test_comment_only_lines_ignored(self):
        """Comments after the last expression shouldn't break anything."""
        code = "x = 5\nx + 3\n# this is a comment"
        # Comment is not a valid ast.Expr, but ast ignores it.
        # The last AST node is still x + 3
        assert _auto_display(code) == "x = 5\nprint(x + 3)\n# this is a comment"

