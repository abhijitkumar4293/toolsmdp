"""Build difficulty buckets from Step 2.2 rollout data.

Classifies each question by avg tool calls across rollouts, computes per-bucket
accuracy, and outputs a structured JSON for use in training and evaluation.

Usage:
    python -m analysis.build_difficulty_buckets
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# Latest v2 result files
RESULT_FILES = {
    "gsm8k": "downloads/vllm_trial/artifacts/outputs/gsm8k_rollout_stats.jsonl",
    "hotpotqa": "downloads/search_v2/artifacts/outputs/hotpotqa_rollout_stats.jsonl",
    "2wiki": "downloads/search_v2/artifacts/outputs/2wiki_rollout_stats.jsonl",
    "finqa": "downloads/finqa_v2/artifacts/outputs/finqa_rollout_stats.jsonl",
    "musique": "downloads/search_v2/artifacts/outputs/musique_rollout_stats.jsonl",
}

OUTPUT_PATH = Path("data_local/analysis/difficulty_buckets.json")


def classify_bucket(avg_tool_calls: float) -> str:
    if avg_tool_calls < 1.5:
        return "1_call"
    elif avg_tool_calls < 2.5:
        return "2_calls"
    else:
        return "3+_calls"


def process_dataset(name: str, path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]

    buckets = {}
    for ex in examples:
        bucket = classify_bucket(ex["avg_tool_calls"])
        if bucket not in buckets:
            buckets[bucket] = {
                "question_indices": [],
                "correct": 0,
                "total": 0,
                "avg_tool_calls_sum": 0.0,
            }
        b = buckets[bucket]
        b["question_indices"].append(ex["question_idx"])
        b["total"] += 1
        b["avg_tool_calls_sum"] += ex["avg_tool_calls"]
        if ex["any_correct"]:
            b["correct"] += 1

    # Compute summary stats
    result = {}
    for bucket_name in ["1_call", "2_calls", "3+_calls"]:
        if bucket_name not in buckets:
            continue
        b = buckets[bucket_name]
        result[bucket_name] = {
            "n": b["total"],
            "pct_of_dataset": round(b["total"] / len(examples), 3),
            "base_accuracy": round(b["correct"] / b["total"], 4) if b["total"] > 0 else 0,
            "correct": b["correct"],
            "avg_tool_calls": round(b["avg_tool_calls_sum"] / b["total"], 2),
            "question_indices": b["question_indices"],
        }

    return result


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_buckets = {}

    print("Building difficulty buckets from Step 2.2 v2 rollouts\n")

    for name, path in RESULT_FILES.items():
        if not Path(path).exists():
            print(f"  {name}: SKIPPED (file not found)")
            continue

        buckets = process_dataset(name, path)
        all_buckets[name] = buckets

        print(f"  {name}:")
        for bname in ["1_call", "2_calls", "3+_calls"]:
            if bname not in buckets:
                continue
            b = buckets[bname]
            print(f"    {bname}: n={b['n']} ({b['pct_of_dataset']:.0%}), "
                  f"EM={b['correct']}/{b['n']} ({b['base_accuracy']:.1%}), "
                  f"avg_tools={b['avg_tool_calls']}")
        print()

    # Summary table
    print("=" * 70)
    print(f"{'Dataset':<12} {'1_call':<20} {'2_calls':<20} {'3+_calls':<20}")
    print("-" * 70)
    for name in RESULT_FILES:
        if name not in all_buckets:
            continue
        parts = []
        for bname in ["1_call", "2_calls", "3+_calls"]:
            if bname in all_buckets[name]:
                b = all_buckets[name][bname]
                parts.append(f"{b['n']:>3}q {b['base_accuracy']:>5.1%}")
            else:
                parts.append(f"{'—':>9}")
        print(f"{name:<12} {parts[0]:<20} {parts[1]:<20} {parts[2]:<20}")
    print("=" * 70)

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        # Don't save question_indices in the summary (too large for readable JSON)
        summary = {}
        for name, buckets in all_buckets.items():
            summary[name] = {}
            for bname, bdata in buckets.items():
                summary[name][bname] = {k: v for k, v in bdata.items() if k != "question_indices"}
        json.dump(summary, f, indent=2)

    # Also save full version with indices
    full_path = OUTPUT_PATH.with_name("difficulty_buckets_full.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(all_buckets, f, indent=2)

    print(f"\nSaved summary to {OUTPUT_PATH}")
    print(f"Saved full (with question indices) to {full_path}")


if __name__ == "__main__":
    main()
