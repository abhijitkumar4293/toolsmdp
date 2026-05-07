"""Detect ```python ... ``` code blocks during generation.

Two interfaces:
  - detect_code_block(text)        : one-shot regex extraction
  - CodeBlockWatcher.feed_token(t) : online state machine for streaming generation
"""
import re
from dataclasses import dataclass


@dataclass
class CodeBlockDetection:
    code: str
    comments: list[str]
    executable: str
    start: int
    end: int


_CODE_BLOCK_PATTERN = re.compile(r"(```(?:python|py|Python)?\s*\n)(.*?)(```)", re.DOTALL)


def detect_code_block(text: str) -> CodeBlockDetection | None:
    m = _CODE_BLOCK_PATTERN.search(text)
    if m is None:
        return None
    body = m.group(2)
    comments, executable_lines = [], []
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("#") and not executable_lines:
            comments.append(s)
        else:
            executable_lines.append(line)
    return CodeBlockDetection(
        code=body.strip(),
        comments=comments,
        executable="\n".join(executable_lines).strip(),
        start=m.start(),
        end=m.end(),
    )


class CodeBlockWatcher:
    """State machine that fires `code_block_complete` when a closing fence is seen."""

    def __init__(self, eos_token: str = "<|endoftext|>"):
        self.state = "NORMAL"
        self.buffer = ""
        self.eos_token = eos_token

    def reset(self):
        self.state = "NORMAL"
        self.buffer = ""

    def feed_token(self, token_text: str) -> str:
        self.buffer += token_text
        if self.eos_token in self.buffer:
            return "eos"
        if self.state == "NORMAL":
            if re.search(r"```(?:python|py|Python)?\s*\n", self.buffer):
                self.state = "IN_CODE_BLOCK"
            return "continue"
        # IN_CODE_BLOCK
        parts = re.split(r"```(?:python|py|Python)?\s*\n", self.buffer, maxsplit=1)
        if len(parts) == 2 and re.search(r"\n\s*```", parts[1]):
            self.state = "NORMAL"
            return "code_block_complete"
        return "continue"

    def get_detection(self) -> CodeBlockDetection | None:
        return detect_code_block(self.buffer)
