"""Tier v2 summary statistics."""
import json
from pathlib import Path

TIER_DIR = Path("downloads/tier_v2/artifacts/outputs")
DATASETS = ["gsm8k", "hotpotqa", "2wiki", "finqa", "musique"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


print("TIER V2 SUMMARY (5000 samples per dataset)")
print("-" * 90)
hdr = "{:<12} {:>6} {:>12} {:>12} {:>20} {:>15} {:>12}"
print(hdr.format("Dataset", "N", "Tier1", "Tier2", "T1_tool_success", "T2_no-tool_ok", "T1_avg_calls"))
print("-" * 90)

for ds in DATASETS:
    rows = load_jsonl(TIER_DIR / f"{ds}_tier_splits.jsonl")
    t1 = [r for r in rows if r["tier"] == "tier1"]
    t2 = [r for r in rows if r["tier"] == "tier2"]
    t1_success = sum(1 for r in t1 if r.get("any_tool_correct"))
    t2_correct = sum(1 for r in t2 if r.get("any_no_tool_correct"))
    avg_calls_t1 = sum(r.get("avg_tool_calls", 0) for r in t1) / len(t1) if t1 else 0

    t1_str = "{} ({:.0f}%)".format(len(t1), len(t1) / len(rows) * 100)
    t2_str = "{} ({:.0f}%)".format(len(t2), len(t2) / len(rows) * 100)
    t1s_str = "{}/{} ({:.1f}%)".format(t1_success, len(t1), t1_success / len(t1) * 100) if t1 else "N/A"
    t2c_str = "{}/{} ({:.1f}%)".format(t2_correct, len(t2), t2_correct / len(t2) * 100) if t2 else "N/A"
    print(hdr.format(ds, len(rows), t1_str, t2_str, t1s_str, t2c_str, "{:.2f}".format(avg_calls_t1)))

print()
print("SEGMENT STRUCTURE (Tier1, tool rollouts only)")
print("-" * 70)
print("{:<12} {:>12} {:>14} {:>14} {:>14}".format(
    "Dataset", "invoke%", "assimilate%", "synthesize%", "avg_segs"))
print("-" * 70)

for ds in DATASETS:
    rows = load_jsonl(TIER_DIR / f"{ds}_tier_splits.jsonl")
    t1 = [r for r in rows if r["tier"] == "tier1"]
    seg_counts = {"invoke": 0, "assimilate": 0, "synthesize": 0}
    total_segs = 0
    total_rollouts = 0
    for r in t1:
        for ro in r.get("tool_rollouts", []):
            for seg in ro.get("segments", []):
                seg_counts[seg.get("type", "?")] = seg_counts.get(seg.get("type", "?"), 0) + 1
                total_segs += 1
            total_rollouts += 1
    if total_segs:
        print("{:<12} {:>12.1f} {:>14.1f} {:>14.1f} {:>14.2f}".format(
            ds,
            seg_counts.get("invoke", 0) / total_segs * 100,
            seg_counts.get("assimilate", 0) / total_segs * 100,
            seg_counts.get("synthesize", 0) / total_segs * 100,
            total_segs / total_rollouts if total_rollouts else 0,
        ))

print()
print("SEARCH QUALITY (Tier1, successful tool rollouts)")
print("-" * 70)
for ds in ["hotpotqa", "2wiki", "musique"]:
    rows = load_jsonl(TIER_DIR / f"{ds}_tier_splits.jsonl")
    t1 = [r for r in rows if r["tier"] == "tier1"]
    empty_searches = 0
    nonempty_searches = 0
    total_searches = 0
    for r in t1:
        for ro in r.get("tool_rollouts", []):
            for seg in ro.get("segments", []):
                if seg.get("type") == "invoke":
                    out = seg.get("output", "")
                    total_searches += 1
                    if "ERROR" in out or out.strip() == "" or "Code produced no output" in out:
                        empty_searches += 1
                    else:
                        nonempty_searches += 1
    print("{}: {}/{} searches returned output ({:.1f}%) | {}/{} errors ({:.1f}%)".format(
        ds, nonempty_searches, total_searches,
        nonempty_searches / total_searches * 100 if total_searches else 0,
        empty_searches, total_searches,
        empty_searches / total_searches * 100 if total_searches else 0,
    ))
