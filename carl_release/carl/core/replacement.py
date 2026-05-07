"""Two-phase replacement (paper Section 3.1 / Appendix B).

Phase 1: code block -> stdout (after invoke segment).
Phase 2: raw stdout -> <context>distilled</context> (after assimilate segment).
"""
from carl.core.code_block_detector import CodeBlockDetection


def replace_code_block(text: str, det: CodeBlockDetection, stdout: str) -> str:
    parts = []
    if det.comments:
        parts.append("\n".join(det.comments))
    if stdout:
        parts.append(stdout)
    return text[:det.start] + "\n".join(parts) + text[det.end:]


def replace_tool_output_with_context(text: str, tool_output: str, context_content: str) -> str:
    idx = text.find(tool_output)
    if idx == -1:
        return text
    return text[:idx] + context_content + text[idx + len(tool_output):]
