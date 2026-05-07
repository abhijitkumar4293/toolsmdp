from carl.core.replacement import replace_code_block, replace_tool_output_with_context
from carl.core.code_block_detector import detect_code_block


def test_replace_code_block_keeps_comments():
    text = "Reasoning.\n```python\n# compute sum\nprint(2+2)\n```\nDone."
    d = detect_code_block(text)
    out = replace_code_block(text, d, "4")
    assert "compute sum" in out
    assert "4" in out
    assert "```" not in out


def test_phase2_replacement():
    text = "Q?\n[TOOL OUTPUT]\nraw stuff\n<context>distilled</context>\n"
    out = replace_tool_output_with_context(text, "[TOOL OUTPUT]\nraw stuff\n", "<context>distilled</context>\n")
    assert "raw stuff" not in out
    assert "<context>distilled</context>" in out
