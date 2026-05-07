"""Segment & Trajectory dataclasses."""
from dataclasses import dataclass, field


@dataclass
class Segment:
    """One option in the SMDP. The model only sees `start_context`."""
    start_context: str
    generated_text: str
    generated_ids: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    segment_type: str = "synthesize"   # invoke | assimilate | synthesize
    termination: str = "eos"           # tool_call | context_block | eos | truncated
    tool_code: str | None = None
    tool_comments: list[str] | None = None
    tool_output: str | None = None
    advantage: float | None = None
    value_estimate: float | None = None
    value_target: float | None = None


@dataclass
class Trajectory:
    segments: list[Segment] = field(default_factory=list)
    full_context: str = ""
    reward: float | None = None

    @property
    def total_tool_calls(self) -> int:
        return sum(1 for s in self.segments if s.segment_type == "invoke")

    @property
    def total_assimilations(self) -> int:
        return sum(1 for s in self.segments if s.segment_type == "assimilate")

    @property
    def hit_segment_limit(self) -> bool:
        return bool(self.segments) and self.segments[-1].termination == "truncated"

    @property
    def num_segments(self) -> int:
        return len(self.segments)
