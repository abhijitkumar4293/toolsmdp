"""Validate tier v3 results against tier v2 and check fixes worked."""
import json
from pathlib import Path
from collections import Counter

V2_DIR = Path("downloads/tier_v2/artifacts/outputs")
V3_DIR = Path("downloads/tier_v3/artifacts/outputs")
DATASETS = ["gsm8k", "hotpotqa", "2wiki", "finqa", "musique"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ── 1. Tier classification comparison ────────────────────────────────────────
print("=" * 70)
print("1. TIER CLASSIFICATION: v2 vs v3")
print("=" * 70)
print(f"{'Dataset':<12} {'v2 T1':>8} {'v3 T1':>8} {'delta':>7}  {'v2 T1_success':>15} {'v3 T1_success':>15}")
print("-" * 70)
for ds in DATASETS:
    v2 = load_jsonl(V2_DIR / f"{ds}_tier_splits.jsonl")
    v3 = load_jsonl(V3_DIR / f"{ds}_tier_splits.jsonl")
    v2_t1 = sum(1 for r in v2 if r["tier"] == "tier1")
    v3_t1 = sum(1 for r in v3 if r["tier"] == "tier1")
    v2_suc = sum(1 for r in v2 if r["any_tool_correct"])
    v3_suc = sum(1 for r in v3 if r["any_tool_correct"])
    n = len(v3)
    print(f"{ds:<12} {v2_t1:>7} ({v2_t1/len(v2)*100:.0f}%) {v3_t1:>7} ({v3_t1/n*100:.0f}%) {v3_t1-v2_t1:>+6}  "
          f"{v2_suc:>6}/{len(v2)} ({v2_suc/len(v2)*100:.1f}%)  {v3_suc:>6}/{n} ({v3_suc/n*100:.1f}%)")

# ── 2. Segment structure ──────────────────────────────────────────────────────
print()
print("=" * 70)
print("2. SEGMENT STRUCTURE: are all 3 types present?")
print("=" * 70)
print(f"{'Dataset':<12} {'invoke%':>9} {'assimilate%':>13} {'synthesize%':>13} {'avg_segs':>9} {'total_rollouts':>15}")
print("-" * 70)
for ds in DATASETS:
    rows = load_jsonl(V3_DIR / f"{ds}_tier_splits.jsonl")
    seg_counts = Counter()
    total_segs = 0
    total_rollouts = 0
    for r in rows:
        for ro in r.get("tool_rollouts", []):
            for seg in ro.get("segments", []):
                seg_counts[seg["type"]] += 1
                total_segs += 1
            total_rollouts += 1
    avg = total_segs / total_rollouts if total_rollouts else 0
    inv = seg_counts["invoke"] / total_segs * 100 if total_segs else 0
    ass = seg_counts["assimilate"] / total_segs * 100 if total_segs else 0
    syn = seg_counts["synthesize"] / total_segs * 100 if total_segs else 0
    print(f"{ds:<12} {inv:>8.1f}% {ass:>12.1f}% {syn:>12.1f}% {avg:>9.2f} {total_rollouts:>15,}")

# ── 3. Context text quality ───────────────────────────────────────────────────
print()
print("=" * 70)
print("3. ASSIMILATE QUALITY: context_block vs eos termination")
print("=" * 70)
for ds in DATASETS:
    rows = load_jsonl(V3_DIR / f"{ds}_tier_splits.jsonl")
    cb, eos, empty = 0, 0, 0
    for r in rows:
        for ro in r.get("tool_rollouts", []):
            for seg in ro.get("segments", []):
                if seg["type"] == "assimilate":
                    t = seg.get("termination", "eos")
                    ct = seg.get("context_text", "").strip()
                    if t == "context_block":
                        cb += 1
                    else:
                        eos += 1
                    if not ct:
                        empty += 1
    total = cb + eos
    print(f"{ds:<12} context_block={cb:>6} ({cb/total*100:.0f}%)  eos={eos:>6} ({eos/total*100:.0f}%)  empty_text={empty}")

# ── 4. Search error rate ──────────────────────────────────────────────────────
print()
print("=" * 70)
print("4. SEARCH ERROR RATE (invoke segments)")
print("=" * 70)
for ds in ["hotpotqa", "2wiki", "musique"]:
    rows = load_jsonl(V3_DIR / f"{ds}_tier_splits.jsonl")
    total, errors = 0, 0
    for r in rows:
        for ro in r.get("tool_rollouts", []):
            for seg in ro.get("segments", []):
                if seg["type"] == "invoke":
                    total += 1
                    if "ERROR" in seg.get("output", ""):
                        errors += 1
    print(f"{ds:<12} {errors:>6}/{total} errors ({errors/total*100:.1f}%)")

# ── 5. Rollout data size ──────────────────────────────────────────────────────
print()
print("=" * 70)
print("5. ROLLOUT DATA SIZE (for critic warmup)")
print("=" * 70)
total_rollouts = 0
total_pairs = 0
for ds in DATASETS:
    rows = load_jsonl(V3_DIR / f"{ds}_tier_splits.jsonl")
    n_rollouts = sum(len(r.get("tool_rollouts", [])) for r in rows)
    n_segs = sum(len(ro.get("segments", [])) for r in rows for ro in r.get("tool_rollouts", []))
    r1 = sum(1 for r in rows for ro in r.get("tool_rollouts", []) if ro.get("reward", 0) > 0)
    r0 = n_rollouts - r1
    total_rollouts += n_rollouts
    total_pairs += n_segs
    fsize = (V3_DIR / f"{ds}_tier_splits.jsonl").stat().st_size / 1e6
    print(f"{ds:<12} {n_rollouts:>8,} rollouts  {n_segs:>9,} seg-states  R=1:{r1:>6,} R=0:{r0:>6,}  {fsize:.0f}MB")
print(f"{'TOTAL':<12} {total_rollouts:>8,} rollouts  {total_pairs:>9,} seg-states")

# ── 6. One good example ───────────────────────────────────────────────────────
print()
print("=" * 70)
print("6. EXAMPLE: invoke → assimilate → synthesize (HotpotQA, correct)")
print("=" * 70)
rows = load_jsonl(V3_DIR / "hotpotqa_tier_splits.jsonl")
for r in rows:
    for ro in r.get("tool_rollouts", []):
        segs = ro.get("segments", [])
        types = [s["type"] for s in segs]
        if ro.get("reward", 0) > 0 and "assimilate" in types and segs[0].get("termination") != "truncated":
            print(f"Q: {r['question'][:120]}")
            print(f"Gold: {r['gold_answer']}  |  Pred: {ro['prediction']}")
            print(f"Segments: {types}")
            for i, seg in enumerate(segs):
                print(f"\n  [Seg {i} — {seg['type'].upper()}]")
                if seg["type"] == "invoke":
                    print(f"    code: {seg.get('code','')[:100].strip()!r}")
                    print(f"    output: {seg.get('output','')[:120].strip()!r}")
                elif seg["type"] == "assimilate":
                    print(f"    termination: {seg.get('termination')}")
                    print(f"    context_text: {seg.get('context_text','')[:150].strip()!r}")
                elif seg["type"] == "synthesize":
                    print(f"    (final answer generated)")
            break
    else:
        continue
    break
