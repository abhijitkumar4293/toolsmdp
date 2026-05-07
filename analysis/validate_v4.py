"""
Validate eval-v4 results against the three fixes:
1. Assimilate segments now present (stop strings working)
2. Search errors lower (print() enforced)
3. Context accumulates (no hallucinated tool outputs)
4. Accuracy comparison vs v3
"""
import json
import re
from pathlib import Path
from collections import Counter

V4_DIR = Path("downloads/eval_v4/artifacts/outputs")
V3_DIR = Path("downloads/eval_v3/artifacts/outputs")
DATASETS = ["gsm8k", "hotpotqa", "2wiki"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ── 1. Accuracy comparison v3 vs v4 ─────────────────────────────────────────
print("=" * 65)
print("1. ACCURACY: v3 (broken loop) vs v4 (fixed loop)")
print("=" * 65)
print(f"{'Dataset':<12} {'v3 pass@4':>10} {'v4 pass@4':>10} {'change':>8}  {'v3 calls':>9} {'v4 calls':>9}")
print("-" * 65)
for ds in DATASETS:
    v3 = load_jsonl(V3_DIR / f"{ds}_rollout_stats.jsonl")
    v4 = load_jsonl(V4_DIR / f"{ds}_rollout_stats.jsonl")
    v3_acc = sum(1 for r in v3 if r["any_correct"]) / len(v3) * 100
    v4_acc = sum(1 for r in v4 if r["any_correct"]) / len(v4) * 100
    v3_calls = sum(r["avg_tool_calls"] for r in v3) / len(v3)
    v4_calls = sum(r["avg_tool_calls"] for r in v4) / len(v4)
    change = v4_acc - v3_acc
    print(f"{ds:<12} {v3_acc:>9.1f}% {v4_acc:>9.1f}% {change:>+7.1f}pp  {v3_calls:>9.2f} {v4_calls:>9.2f}")

# ── 2. Segment type breakdown ────────────────────────────────────────────────
print()
print("=" * 65)
print("2. SEGMENT TYPES: are assimilate segments now present?")
print("=" * 65)
print(f"{'Dataset':<12} {'invoke%':>9} {'assimilate%':>12} {'synthesize%':>13} {'avg_segs':>9}")
print("-" * 65)
for ds in DATASETS:
    rows = load_jsonl(V4_DIR / f"{ds}_rollout_stats.jsonl")
    seg_counts = Counter()
    total_segs = 0
    total_rollouts = 0
    for r in rows:
        for ro in r["rollouts"]:
            for seg in ro.get("segments", []):
                seg_counts[seg["type"]] += 1
                total_segs += 1
            total_rollouts += 1
    avg_segs = total_segs / total_rollouts if total_rollouts else 0
    inv = seg_counts["invoke"] / total_segs * 100 if total_segs else 0
    ass = seg_counts["assimilate"] / total_segs * 100 if total_segs else 0
    syn = seg_counts["synthesize"] / total_segs * 100 if total_segs else 0
    print(f"{ds:<12} {inv:>8.1f}% {ass:>11.1f}% {syn:>12.1f}% {avg_segs:>9.2f}")

# ── 3. Search error rate ─────────────────────────────────────────────────────
print()
print("=" * 65)
print("3. SEARCH ERRORS: invoke segments with ERROR output")
print("=" * 65)
for ds in ["hotpotqa", "2wiki"]:
    v3 = load_jsonl(V3_DIR / f"{ds}_rollout_stats.jsonl")
    v4 = load_jsonl(V4_DIR / f"{ds}_rollout_stats.jsonl")
    def error_rate(rows):
        total, errors = 0, 0
        for r in rows:
            for ro in r["rollouts"]:
                for seg in ro.get("segments", []):
                    if seg["type"] == "invoke":
                        total += 1
                        if "ERROR" in seg.get("output", ""):
                            errors += 1
        return errors, total
    v3e, v3t = error_rate(v3)
    v4e, v4t = error_rate(v4)
    print(f"{ds}: v3 {v3e}/{v3t} errors ({v3e/v3t*100:.1f}%)  →  v4 {v4e}/{v4t} errors ({v4e/v4t*100:.1f}%)")

# ── 4. Context block quality ─────────────────────────────────────────────────
print()
print("=" * 65)
print("4. ASSIMILATE QUALITY: what do <context> blocks contain?")
print("=" * 65)
for ds in ["hotpotqa", "2wiki"]:
    rows = load_jsonl(V4_DIR / f"{ds}_rollout_stats.jsonl")
    examples = []
    skipped = 0
    for r in rows:
        for ro in r["rollouts"]:
            for seg in ro.get("segments", []):
                if seg["type"] == "assimilate":
                    ct = seg.get("context_text", "").strip()
                    if ct and len(examples) < 3:
                        examples.append((r["question"][:80], seg.get("raw_stdout","")[:120], ct))
                    elif not ct:
                        skipped += 1
    print(f"\n{ds} — assimilate segments with empty context_text: {skipped}")
    for q, raw, ctx in examples:
        print(f"  Q: {q}")
        print(f"  raw_stdout: {raw[:100]!r}")
        print(f"  context_text: {ctx[:150]!r}")
        print()

# ── 5. Full rollout example: correct with real search ────────────────────────
print("=" * 65)
print("5. EXAMPLE: correct rollout showing invoke→assimilate→synthesize")
print("=" * 65)
for ds in ["hotpotqa", "2wiki"]:
    rows = load_jsonl(V4_DIR / f"{ds}_rollout_stats.jsonl")
    found = False
    for r in rows:
        if found:
            break
        for ro in r["rollouts"]:
            segs = ro.get("segments", [])
            types = [s["type"] for s in segs]
            if ro["reward"] > 0 and "assimilate" in types:
                print(f"\n{ds.upper()} — Q: {r['question'][:120]}")
                print(f"Gold: {r['gold_answer']}  |  Pred: {ro['prediction']}  |  Segs: {types}")
                print(ro.get("text", ro.get("full_generated", ""))[:1500])
                found = True
                break

# ── 6. Bucket breakdown v4 ───────────────────────────────────────────────────
print()
print("=" * 65)
print("6. DIFFICULTY BUCKETS v4")
print("=" * 65)
for ds in DATASETS:
    rows = load_jsonl(V4_DIR / f"{ds}_rollout_stats.jsonl")
    buckets = {}
    for r in rows:
        b = r["bucket"]
        if b not in buckets:
            buckets[b] = [0, 0]
        buckets[b][0] += 1
        buckets[b][1] += int(r["any_correct"])
    bstr = "  ".join(f"{b}: {v[1]}/{v[0]} ({v[1]/v[0]*100:.0f}%)" for b, v in sorted(buckets.items()))
    print(f"{ds:<12}  {bstr}")
