from carl.core.context_block_detector import detect_context_block, ContextBlockWatcher


def test_detect_context():
    d = detect_context_block("blah <context>fact A; fact B</context> rest")
    assert d.content == "fact A; fact B"


def test_no_context():
    assert detect_context_block("no tag") is None


def test_watcher_signals_completion():
    w = ContextBlockWatcher()
    for t in ["<context>", "fact", "</context>"]:
        sig = w.feed_token(t)
    assert sig == "context_block_complete"


def test_watcher_budget():
    w = ContextBlockWatcher(max_tokens=2)
    w.feed_token("<context>")
    w.feed_token("a")
    sig = w.feed_token("b")
    assert sig in ("budget_exceeded", "continue")
