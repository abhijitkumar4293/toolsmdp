"""Tier 1 / Tier 2 labeling via 5-rollout no-tool consistency (paper Section 4.1, Appendix C).

A question is Tier 2 (within parametric competence) if the base model answers it
correctly in at least one of N (=5) independent no-tool rollouts. Otherwise Tier 1.

Writes a labels JSONL: {idx, dataset, tier, n_correct_no_tool, n_total}.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Iterable

from carl.core.reward import compute_reward


NO_TOOL_PROMPT = ("Answer the following question directly. Do not use any tools or code. "
                  "Provide your answer inside <answer></answer>.")


def classify_tier(n_correct: int) -> str:
    return "tier2" if n_correct >= 1 else "tier1"


def build_tier_splits(
    rows: Iterable[dict],
    sample_no_tool: Callable[[str], str],   # (question) -> generated answer text
    n_rollouts: int = 5,
    out_path: str | Path | None = None,
) -> list[dict]:
    """Run N no-tool rollouts per question, label tier, optionally write JSONL."""
    labels = []
    for r in rows:
        q = f"{NO_TOOL_PROMPT}\n\nQuestion: {r['question']}"
        n_ok = 0
        for _ in range(n_rollouts):
            text = sample_no_tool(q)
            n_ok += int(compute_reward(text, r["gold"], r["dataset"]) == 1.0)
        labels.append({
            "idx": r["idx"], "dataset": r["dataset"],
            "tier": classify_tier(n_ok),
            "n_correct_no_tool": n_ok, "n_total": n_rollouts,
        })
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for l in labels:
                f.write(json.dumps(l) + "\n")
    return labels
