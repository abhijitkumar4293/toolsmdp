"""Tool-absence resilience eval (paper Appendix I).

Force CARL to invoke a tool on Tier-2 questions; replace stdout with one of:
    - working retrieval (control)
    - empty
    - irrelevant passage
    - malformed garbage
Measure EM at each condition.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable

from carl.core.rollout import rollout, CARL_SYSTEM_PROMPT
from carl.core.reward import compute_reward


_FORCE_TOOL = CARL_SYSTEM_PROMPT + "\n\nYOU MUST USE A TOOL ONCE. Do not skip the code block.\n"

_GARBAGE = "%@!#xq qz!@# fhquw 8923hf !@#$ ahsdf"
_IRRELEVANT = "[1] Wikipedia: Eclipse (software).\nEclipse is an integrated development environment used in computer programming."


def _make_force_executor(real_exec, mode: str):
    def fn(code: str) -> str:
        out = real_exec(code)
        if mode == "working": return out
        if mode == "empty":   return ""
        if mode == "irrelevant": return _IRRELEVANT
        if mode == "garbage": return _GARBAGE
        raise ValueError(mode)
    return fn


def run_tool_resilience(tier2_prompts, generate: Callable, real_execute: Callable,
                        out_dir: str) -> dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results = {}
    for mode in ("working", "empty", "irrelevant", "garbage"):
        exec_fn = _make_force_executor(real_execute, mode)
        n_ok = 0
        for ex in tier2_prompts:
            tr = rollout(ex["question"], generate, exec_fn, system_prompt=_FORCE_TOOL)
            n_ok += int(compute_reward(tr.full_context, ex["gold"], ex["dataset"]) == 1.0)
        results[mode] = {"EM": n_ok / len(tier2_prompts), "n": len(tier2_prompts)}
    Path(out_dir, "resilience.json").write_text(json.dumps(results, indent=2))
    return results
