"""Full evaluation harness: rolls out, computes EM, tool calls, tokens, V(s_0).

Outputs a JSONL file `predictions.jsonl` (one row per (q, rollout)) and a
summary JSON `metrics.json` (EM, avg tool calls, avg tokens).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable
from collections import defaultdict

from carl.core.rollout import rollout
from carl.core.reward import compute_reward, extract_answer


def evaluate(prompts, generate: Callable, execute: Callable,
             out_dir: str, tag: str, critic_eval: Callable | None = None,
             **rollout_kw) -> dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pred_path = Path(out_dir) / f"{tag}_predictions.jsonl"
    metrics_path = Path(out_dir) / f"{tag}_metrics.json"

    by_ds = defaultdict(lambda: {"n": 0, "n_correct": 0, "tool_calls": 0,
                                 "tokens": 0, "v_s0": 0.0})
    with open(pred_path, "w") as f:
        for ex in prompts:
            tr = rollout(ex["question"], generate, execute, **rollout_kw)
            tr.reward = compute_reward(tr.full_context, ex["gold"], ex["dataset"])
            v0 = critic_eval(tr.segments[0].start_context) if critic_eval and tr.segments else 0.0
            row = {
                "idx": ex["idx"], "dataset": ex["dataset"],
                "question": ex["question"], "gold": ex["gold"],
                "prediction": extract_answer(tr.full_context, ex["dataset"]),
                "reward": tr.reward, "n_tool_calls": tr.total_tool_calls,
                "n_tokens": sum(len(s.generated_ids) for s in tr.segments),
                "n_segments": tr.num_segments, "v_s0": v0,
                "tier": ex.get("tier"), "n_hops": ex.get("n_hops"),
            }
            f.write(json.dumps(row) + "\n")
            d = by_ds[ex["dataset"]]
            d["n"] += 1; d["n_correct"] += int(tr.reward == 1.0)
            d["tool_calls"] += tr.total_tool_calls; d["tokens"] += row["n_tokens"]
            d["v_s0"] += v0

    summary = {}
    for ds, d in by_ds.items():
        summary[ds] = {
            "EM": d["n_correct"] / d["n"], "n": d["n"],
            "avg_tool_calls": d["tool_calls"] / d["n"],
            "avg_tokens": d["tokens"] / d["n"],
            "avg_v_s0": d["v_s0"] / d["n"],
        }
    summary["overall"] = {
        "EM": sum(d["n_correct"] for d in by_ds.values()) / sum(d["n"] for d in by_ds.values()),
        "n": sum(d["n"] for d in by_ds.values()),
    }
    metrics_path.write_text(json.dumps(summary, indent=2))
    return summary
