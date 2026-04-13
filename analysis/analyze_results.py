"""
Quick analysis of eval_v3 and tier_v2 results.
Usage: python analysis/analyze_results.py
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

EVAL_DIR = Path("downloads/eval_v3/artifacts/outputs")
TIER_DIR = Path("downloads/tier_v2/artifacts/outputs")
DATASETS = ["gsm8k", "hotpotqa", "2wiki", "finqa", "musique"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ── 1. Eval v3 summary ───────────────────────────────────────────────────────

def eval_summary():
    print("=" * 60)
    print("EVAL V3 SUMMARY (500 samples, pass@4)")
    print("=" * 60)
    print(f"{'Dataset':<12} {'N':>5} {'pass@4':>8} {'avg_calls':>10} {'bucket breakdown'}")
    print("-" * 60)
    for ds in DATASETS:
        rows = load_jsonl(EVAL_DIR / f"{ds}_rollout_stats.jsonl")
        total = len(rows)
        correct = sum(1 for r in rows if r.get("any_correct"))
        avg_calls = sum(r.get("avg_tool_calls", 0) for r in rows) / total
        buckets = defaultdict(lambda: [0, 0])
        for r in rows:
            b = r.get("bucket", "?")
            buckets[b][0] += 1
            buckets[b][1] += int(r.get("any_correct", False))
        bstr = "  ".join(
            f"{b}:{v[1]}/{v[0]}"
            for b, v in sorted(buckets.items())
        )
        print(f"{ds:<12} {total:>5} {correct/total*100:>7.1f}% {avg_calls:>10.2f}  {bstr}")


# ── 2. Concrete examples: correct search rollouts ────────────────────────────

def show_examples(ds, n_correct=2, n_wrong=2):
    rows = load_jsonl(EVAL_DIR / f"{ds}_rollout_stats.jsonl")
    correct = [r for r in rows if r.get("any_correct")]
    wrong = [r for r in rows if not r.get("any_correct")]

    print(f"\n{'=' * 60}")
    print(f"EXAMPLES — {ds.upper()}")
    print(f"{'=' * 60}")

    print(f"\n--- CORRECT ({min(n_correct, len(correct))} of {len(correct)}) ---")
    for ex in correct[:n_correct]:
        best = next((ro for ro in ex["rollouts"] if ro["reward"] > 0), ex["rollouts"][0])
        print(f"\nQ: {ex['question'][:300]}")
        print(f"Gold: {ex['gold_answer']}")
        print(f"Pred: {best['prediction']}  | tool_calls={best['num_tool_calls']}  types={best.get('tool_types', [])}")
        gen = best["full_generated"]
        # Show up to 1200 chars
        print(gen[:1200])
        if len(gen) > 1200:
            print(f"  ... [{len(gen) - 1200} more chars]")

    print(f"\n--- WRONG ({min(n_wrong, len(wrong))} of {len(wrong)}) ---")
    for ex in wrong[:n_wrong]:
        worst = ex["rollouts"][0]
        print(f"\nQ: {ex['question'][:300]}")
        print(f"Gold: {ex['gold_answer']}")
        print(f"Pred: {worst['prediction']}  | tool_calls={worst['num_tool_calls']}  types={worst.get('tool_types', [])}")
        gen = worst["full_generated"]
        print(gen[:1200])
        if len(gen) > 1200:
            print(f"  ... [{len(gen) - 1200} more chars]")


# ── 3. Tier v2 summary ───────────────────────────────────────────────────────

def tier_summary():
    print("\n" + "=" * 60)
    print("TIER V2 SUMMARY (5000 samples per dataset)")
    print("=" * 60)
    print(f"{'Dataset':<12} {'N':>6} {'Tier1':>8} {'Tier2':>8} {'tool_success':>14}")
    print("-" * 60)
    for ds in DATASETS:
        t1 = load_jsonl(TIER_DIR / f"{ds}_tier1.jsonl")
        t2 = load_jsonl(TIER_DIR / f"{ds}_tier2.jsonl")
        total = len(t1) + len(t2)
        # tool success: tier1 questions where any tool rollout was correct
        t1_success = sum(1 for r in t1 if r.get("any_tool_correct", False))
        print(f"{ds:<12} {total:>6} {len(t1):>7} ({len(t1)/total*100:.0f}%)  {len(t2):>6} ({len(t2)/total*100:.0f}%)  {t1_success}/{len(t1)} ({t1_success/len(t1)*100:.1f}% of T1)" if t1 else f"{ds:<12} {total:>6} no tier1")


# ── 4. Failure mode analysis ─────────────────────────────────────────────────

def failure_modes(ds):
    rows = load_jsonl(EVAL_DIR / f"{ds}_rollout_stats.jsonl")
    wrong = [r for r in rows if not r.get("any_correct")]
    print(f"\n{'=' * 60}")
    print(f"FAILURE MODES — {ds.upper()} ({len(wrong)} wrong / {len(rows)} total)")
    print(f"{'=' * 60}")

    no_tool = sum(1 for r in wrong if r.get("avg_tool_calls", 0) < 0.5)
    many_calls = sum(1 for r in wrong if r.get("avg_tool_calls", 0) >= 3)
    bucket_counts = defaultdict(int)
    for r in wrong:
        bucket_counts[r.get("bucket", "?")] += 1

    print(f"  No tool calls (avg<0.5): {no_tool}")
    print(f"  3+ tool calls:           {many_calls}")
    print(f"  By bucket: {dict(bucket_counts)}")

    # Sample a "near miss" — gold short, pred close
    near = [r for r in wrong if len(r.get("gold_answer","")) < 30]
    if near:
        ex = near[0]
        ro = ex["rollouts"][0]
        print(f"\n  Near miss example:")
        print(f"  Q: {ex['question'][:200]}")
        print(f"  Gold: {ex['gold_answer']}")
        print(f"  All preds: {[ro2['prediction'] for ro2 in ex['rollouts']]}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    eval_summary()

    if target in ("all", "examples"):
        for ds in ["hotpotqa", "2wiki", "musique", "gsm8k", "finqa"]:
            show_examples(ds, n_correct=2, n_wrong=2)

    if target in ("all", "tiers"):
        tier_summary()

    if target in ("all", "failures"):
        for ds in ["hotpotqa", "2wiki", "musique"]:
            failure_modes(ds)
