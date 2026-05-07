"""Build the 4-bucket warm-up dataset (paper Appendix C).

Each input trajectory has:
  - tier in {"tier1", "tier2"}    (from carl.data.tier_split)
  - prompt_mode in {"no_tool", "forced_tool"}
  - reward                        (binary terminal R)
  - segments[] with segment_type and context_snapshot

Trajectories are bucketed by (tier, prompt_mode), each bucket is subsampled
to the paper's target count for the chosen scale, then one (context,
V_target) pair is emitted per segment of every kept trajectory.

The four buckets play the roles described in Section 3.3 / Appendix C:
  tier2 / no_tool      anchors V(s_0) ~ 1
  tier2 / forced_tool  teaches "unnecessary tool risk"
  tier1 / no_tool      anchors V(s_0) ~ 0
  tier1 / forced_tool  teaches assimilation
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Iterable

# Paper Appendix C: 32K total per scale.
BUCKET_COUNTS_7B = {
    ("tier2", "no_tool"):     7200,
    ("tier2", "forced_tool"): 7200,
    ("tier1", "no_tool"):     8800,
    ("tier1", "forced_tool"): 8800,
}
BUCKET_COUNTS_3B = {
    ("tier2", "no_tool"):     5600,
    ("tier2", "forced_tool"): 5600,
    ("tier1", "no_tool"):    10400,
    ("tier1", "forced_tool"): 10400,
}
BUCKET_COUNTS = {"7b": BUCKET_COUNTS_7B, "3b": BUCKET_COUNTS_3B}


def _bucket_key(tr: dict) -> tuple[str, str] | None:
    t = tr.get("tier"); p = tr.get("prompt_mode")
    if t in ("tier1", "tier2") and p in ("no_tool", "forced_tool"):
        return (t, p)
    return None


def build_warmup_pairs(
    trajectories: Iterable[dict],
    out_path: str | Path,
    scale: str = "3b",
    seed: int = 0,
) -> dict:
    """Bucket, subsample, emit (context, V_target) pairs.

    Returns a stats dict with per-bucket trajectory and pair counts.
    """
    targets = BUCKET_COUNTS[scale.lower()]
    rng = random.Random(seed)

    by_bucket: dict[tuple[str, str], list[dict]] = {k: [] for k in targets}
    for tr in trajectories:
        k = _bucket_key(tr)
        if k in by_bucket:
            by_bucket[k].append(tr)

    selected: dict[tuple[str, str], list[dict]] = {}
    for k, target in targets.items():
        pool = by_bucket[k]
        if len(pool) <= target:
            selected[k] = pool
        else:
            selected[k] = rng.sample(pool, target)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    stats = {"scale": scale,
             "target_per_bucket": {f"{t}/{m}": v for (t, m), v in targets.items()},
             "kept_trajectories": {}, "kept_pairs": {}}
    n_total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for (tier, mode), trs in selected.items():
            stats["kept_trajectories"][f"{tier}/{mode}"] = len(trs)
            n_pairs_bucket = 0
            for tr in trs:
                R = float(tr["reward"])
                for s in tr["segments"]:
                    if s["segment_type"] not in ("invoke", "assimilate", "synthesize"):
                        continue
                    f.write(json.dumps({
                        "context": s["context_snapshot"],
                        "V_target": R,
                        "segment_type": s["segment_type"],
                        "bucket": f"{tier}/{mode}",
                        "tier": tier, "prompt_mode": mode,
                        "dataset": tr["dataset"], "q_idx": tr["q_idx"],
                        "rollout_idx": tr.get("rollout_idx", 0),
                    }) + "\n")
                    n_pairs_bucket += 1
                    n_total += 1
            stats["kept_pairs"][f"{tier}/{mode}"] = n_pairs_bucket
    stats["n_pairs_total"] = n_total
    return stats
