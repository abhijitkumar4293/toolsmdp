"""Hallucination eval (paper Appendix J).

200 Tier-1 questions where the base model fails 5/5 no-tool rollouts. Run four
policies (base no-tool, base selective-tools, Search-R1 PPO, CARL); count
confident-wrong predictions: a non-null answer that is wrong.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable

from carl.core.reward import extract_answer, exact_match


def _confident_wrong_rate(prompts, run_one):
    n_cw, n_total = 0, 0
    for ex in prompts:
        text = run_one(ex)
        pred = extract_answer(text, ex["dataset"])
        if pred is not None and not exact_match(pred, ex["gold"]):
            n_cw += 1
        n_total += 1
    return n_cw / max(n_total, 1)


def run_hallucination_eval(tier1_prompts, policies: dict[str, Callable], out_dir: str):
    """policies = {'base_no_tool': fn(ex)->text, 'sr1_ppo': fn(ex)->text, 'carl': fn(ex)->text, ...}"""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = {name: _confident_wrong_rate(tier1_prompts, fn) for name, fn in policies.items()}
    Path(out_dir, "hallucination.json").write_text(json.dumps(out, indent=2))
    return out
