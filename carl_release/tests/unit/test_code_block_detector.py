from carl.core.code_block_detector import detect_code_block, CodeBlockWatcher


def test_detect_simple():
    text = "Let me compute.\n```python\nprint(2+2)\n```\nDone."
    d = detect_code_block(text)
    assert d is not None
    assert "print(2+2)" in d.executable
    assert d.start < d.end


def test_detect_with_comment():
    text = "```python\n# compute the sum\nprint(2+2)\n```"
    d = detect_code_block(text)
    assert "# compute the sum" in d.comments[0]


def test_no_block():
    assert detect_code_block("no code here") is None


def test_watcher_streams_then_completes():
    w = CodeBlockWatcher()
    for tok in ["```python\n", "print(2+2)\n", "```"]:
        sig = w.feed_token(tok)
    assert sig == "code_block_complete"
