from carl.sandbox.executor import execute_code, extract_search_query_strings


def test_print():
    assert execute_code("print('hello')").strip() == "hello"


def test_auto_display():
    out = execute_code("2+2")
    assert "4" in out


def test_with_search_results():
    code = "x = search('foo')\nprint(x)"
    out = execute_code(code, search_results={"foo": "BAR"})
    assert "BAR" in out


def test_extract_queries():
    qs = extract_search_query_strings('search("a")\nresult = search(\'b\')')
    assert qs == ["a", "b"]
