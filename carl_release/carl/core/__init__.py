"""Core segment plumbing: code/context detectors, two-phase replacement, dataclasses, reward."""
from carl.core.code_block_detector import detect_code_block, CodeBlockDetection, CodeBlockWatcher
from carl.core.context_block_detector import detect_context_block, ContextBlockDetection, ContextBlockWatcher
from carl.core.replacement import replace_code_block, replace_tool_output_with_context
from carl.core.segment import Segment, Trajectory
from carl.core.reward import (
    normalize_answer, extract_answer, exact_match, compute_reward,
)
