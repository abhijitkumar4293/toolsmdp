"""Detect <context>...</context> assimilation blocks."""
import re
from dataclasses import dataclass


@dataclass
class ContextBlockDetection:
    content: str
    start: int
    end: int


_CTX = re.compile(r"<context>(.*?)</context>", re.DOTALL)


def detect_context_block(text: str) -> ContextBlockDetection | None:
    m = _CTX.search(text)
    if m is None:
        return None
    return ContextBlockDetection(content=m.group(1).strip(), start=m.start(), end=m.end())


class ContextBlockWatcher:
    """Fires `context_block_complete` on closing tag, or `budget_exceeded` after T_assim tokens."""

    def __init__(self, eos_token: str = "<|endoftext|>", max_tokens: int = 256):
        self.state = "NORMAL"
        self.buffer = ""
        self.eos_token = eos_token
        self.max_tokens = max_tokens
        self._n = 0

    def reset(self):
        self.state, self.buffer, self._n = "NORMAL", "", 0

    def feed_token(self, token_text: str) -> str:
        self.buffer += token_text
        if self.eos_token in self.buffer:
            return "eos"
        if self.state == "NORMAL":
            if "<context>" in self.buffer:
                self.state = "IN_CONTEXT_BLOCK"
                self._n = 0
            return "continue"
        self._n += 1
        if "</context>" in self.buffer:
            self.state = "NORMAL"
            return "context_block_complete"
        if self._n >= self.max_tokens:
            self.state = "NORMAL"
            return "budget_exceeded"
        return "continue"

    def get_detection(self) -> ContextBlockDetection | None:
        return detect_context_block(self.buffer)
