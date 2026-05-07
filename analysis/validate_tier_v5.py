"""Validate tier v5 data — Step 2.4.2 automated checks.

Six automated checks:
1. Phase-2 replacement: after every assimilate, context_snapshot has no [TOOL OUTPUT]
2. Context shrinks after assimilate vs preceding invoke
3. context_snapshot stored in every segment
4. Final snapshot matches rollout context field
5. Tier classification unchanged from v4 (±5%)
6. No-output rate <5% (auto-display working)

Plus: summary stats, spot-check examples.

Usage:
    python -m analysis.validate_tier_v5 --input-dir downloads/tier_v5/artifacts/outputs
"""

import argparse
import json
from pathlib import Path

DATASETS = ["gsm8k", "hotpotqa", "2wiki", "finqa", "musique"]

# v4 tier classification for comparison (from results.md)
V4_TIERS = {
    "gsm8k":    {"tier1": 151, "tier2": 4849},
    "hotpotqa":  {"tier1": 3802, "tier2": 1198},
    "2wiki":     {"tier1": 2864, "tier2": 2136},
    "finqa":     {"tier1": 4560, "tier2": 440},
    "musique":   {"tier1": 4674, "tier2": 326},
}


def validate_dataset(path: Path, dataset: str) -> dict:
    """Run all 6 checks on one dataset file."""

    stats = {
        "n_questions": 0,
        "tier1": 0, "tier2": 0,
        "n_rollouts": 0, "n_segments": 0,
        "n_tool_rollouts": 0, "n_no_tool_rollouts": 0,
        # Check 1: phase-2 replacement
        "assimilate_with_tool_output": 0,
        "assimilate_total": 0,
        # Check 2: context shrinks
        "shrink_pass": 0, "shrink_fail": 0,
        "shrink_fail_examples": [],
        # Check 3: context_snapshot present
        "snapshot_present": 0, "snapshot_missing": 0,
        # Check 4: final snapshot matches
        "final_match": 0, "final_mismatch": 0,
        "final_mismatch_examples": [],
        # Check 5: tier classification (computed after)
        # Check 6: no-output rate
        "invoke_total": 0, "invoke_no_output": 0,
        # Tool success
        "any_tool_correct": 0,
        # Spot-check examples
        "spot_checks": [],
    }

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            row = json.loads(line)
            stats["n_questions"] += 1
            if row["tier"] == "tier1":
                stats["tier1"] += 1
            else:
                stats["tier2"] += 1

            if row.get("any_tool_correct"):
                stats["any_tool_correct"] += 1

            stats["n_no_tool_rollouts"] += len(row.get("no_tool_rollouts", []))

            for ri, rollout in enumerate(row["tool_rollouts"]):
                stats["n_tool_rollouts"] += 1
                segments = rollout["segments"]
                stats["n_segments"] += len(segments)

                prev_invoke_snapshot_len = None

                for si, seg in enumerate(segments):
                    seg_type = seg["type"]

                    # Check 3: context_snapshot present
                    if "context_snapshot" in seg:
                        stats["snapshot_present"] += 1
                        snapshot = seg["context_snapshot"]
                    else:
                        stats["snapshot_missing"] += 1
                        continue

                    if seg_type == "invoke":
                        stats["invoke_total"] += 1
                        # Check 6: no-output
                        output = seg.get("output", "")
                        if output.strip() == "" or "Code produced no output" in output:
                            stats["invoke_no_output"] += 1
                        prev_invoke_snapshot_len = len(snapshot)

                    elif seg_type == "assimilate":
                        stats["assimilate_total"] += 1

                        # Check 1: no [TOOL OUTPUT] in assimilate snapshot
                        if "[TOOL OUTPUT]" in snapshot:
                            stats["assimilate_with_tool_output"] += 1

                        # Check 2: context shrinks after assimilate
                        if prev_invoke_snapshot_len is not None:
                            if len(snapshot) < prev_invoke_snapshot_len:
                                stats["shrink_pass"] += 1
                            else:
                                stats["shrink_fail"] += 1
                                if len(stats["shrink_fail_examples"]) < 3:
                                    stats["shrink_fail_examples"].append({
                                        "q_idx": row["question_idx"],
                                        "rollout_idx": ri,
                                        "seg_idx": si,
                                        "invoke_len": prev_invoke_snapshot_len,
                                        "assimilate_len": len(snapshot),
                                    })
                        prev_invoke_snapshot_len = None  # reset

                # Check 4: final snapshot matches rollout context
                if segments and "context_snapshot" in segments[-1]:
                    final_snap = segments[-1]["context_snapshot"]
                    rollout_ctx = rollout.get("context", "")
                    if final_snap == rollout_ctx:
                        stats["final_match"] += 1
                    else:
                        stats["final_mismatch"] += 1
                        if len(stats["final_mismatch_examples"]) < 2:
                            stats["final_mismatch_examples"].append({
                                "q_idx": row["question_idx"],
                                "rollout_idx": ri,
                                "snap_len": len(final_snap),
                                "ctx_len": len(rollout_ctx),
                                "snap_tail": final_snap[-200:],
                                "ctx_tail": rollout_ctx[-200:],
                            })

                # Collect spot-check: Tier 1, R=1, 2 tool calls
                if (row["tier"] == "tier1" and rollout["reward"] > 0
                        and rollout["num_tool_calls"] == 2
                        and len(stats["spot_checks"]) < 2):
                    stats["spot_checks"].append({
                        "q_idx": row["question_idx"],
                        "question": row["question"],
                        "reward": rollout["reward"],
                        "rollout_idx": ri,
                        "segments": [
                            {
                                "type": s["type"],
                                "snapshot_len": len(s.get("context_snapshot", "")),
                                "snapshot_preview": s.get("context_snapshot", "")[:500],
                                "has_tool_output_marker": "[TOOL OUTPUT]" in s.get("context_snapshot", ""),
                            }
                            for s in segments
                        ],
                    })

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
                        default=Path("downloads/tier_v5/artifacts/outputs"))
    args = parser.parse_args()

    all_stats = {}
    for ds in DATASETS:
        path = args.input_dir / f"{ds}_tier_splits.jsonl"
        if not path.exists():
            print(f"SKIP: {ds} not found at {path}")
            continue
        print(f"Validating {ds}...")
        all_stats[ds] = validate_dataset(path, ds)

    print("\n" + "=" * 80)
    print("TIER v5 VALIDATION REPORT")
    print("=" * 80)

    # --- Summary stats ---
    print("\n## Summary Stats")
    print(f"{'Dataset':<12} {'N':>6} {'Tier1':>7} {'Tier2':>7} {'ToolRoll':>10} {'Segments':>10} {'ToolSuccess':>12}")
    total_q = total_t1 = total_t2 = total_roll = total_seg = 0
    for ds in DATASETS:
        if ds not in all_stats:
            continue
        s = all_stats[ds]
        pct = f"{100*s['any_tool_correct']/s['n_questions']:.1f}%" if s['n_questions'] else "n/a"
        print(f"{ds:<12} {s['n_questions']:>6} {s['tier1']:>7} {s['tier2']:>7} {s['n_tool_rollouts']:>10} {s['n_segments']:>10} {pct:>12}")
        total_q += s['n_questions']
        total_t1 += s['tier1']
        total_t2 += s['tier2']
        total_roll += s['n_tool_rollouts']
        total_seg += s['n_segments']
    print(f"{'TOTAL':<12} {total_q:>6} {total_t1:>7} {total_t2:>7} {total_roll:>10} {total_seg:>10}")

    # --- Check 1: Phase-2 replacement ---
    print("\n## Check 1: Phase-2 Replacement (no [TOOL OUTPUT] in assimilate snapshots)")
    all_pass = True
    for ds in DATASETS:
        if ds not in all_stats:
            continue
        s = all_stats[ds]
        bad = s["assimilate_with_tool_output"]
        total = s["assimilate_total"]
        pct = f"{100*bad/total:.1f}%" if total else "n/a"
        status = "PASS" if bad == 0 else "FAIL"
        if bad > 0:
            all_pass = False
        print(f"  {ds:<12} {status} — {bad}/{total} assimilate snapshots contain [TOOL OUTPUT] ({pct})")
    print(f"  {'>> OVERALL:':<12} {'PASS' if all_pass else 'FAIL'}")

    # --- Check 2: Context shrinks after assimilate ---
    print("\n## Check 2: Context Shrinks After Assimilate")
    all_pass = True
    for ds in DATASETS:
        if ds not in all_stats:
            continue
        s = all_stats[ds]
        p = s["shrink_pass"]
        f_ = s["shrink_fail"]
        total = p + f_
        pct = f"{100*p/total:.1f}%" if total else "n/a"
        status = "PASS" if f_ == 0 else f"WARN ({f_} failures)"
        if f_ > 0:
            all_pass = False
        print(f"  {ds:<12} {status} — {p}/{total} pairs shrink ({pct})")
        for ex in s["shrink_fail_examples"][:1]:
            print(f"    Example: q{ex['q_idx']} r{ex['rollout_idx']} s{ex['seg_idx']}: invoke={ex['invoke_len']} -> assimilate={ex['assimilate_len']}")
    print(f"  {'>> OVERALL:':<12} {'PASS' if all_pass else 'WARN'}")

    # --- Check 3: context_snapshot stored ---
    print("\n## Check 3: context_snapshot Stored in Every Segment")
    all_pass = True
    for ds in DATASETS:
        if ds not in all_stats:
            continue
        s = all_stats[ds]
        present = s["snapshot_present"]
        missing = s["snapshot_missing"]
        status = "PASS" if missing == 0 else "FAIL"
        if missing > 0:
            all_pass = False
        print(f"  {ds:<12} {status} — {present} present, {missing} missing")
    print(f"  {'>> OVERALL:':<12} {'PASS' if all_pass else 'FAIL'}")

    # --- Check 4: Final snapshot matches rollout context ---
    print("\n## Check 4: Final Snapshot Matches Rollout Context")
    all_pass = True
    for ds in DATASETS:
        if ds not in all_stats:
            continue
        s = all_stats[ds]
        m = s["final_match"]
        mm = s["final_mismatch"]
        total = m + mm
        status = "PASS" if mm == 0 else f"FAIL ({mm} mismatches)"
        if mm > 0:
            all_pass = False
        print(f"  {ds:<12} {status} — {m}/{total} match")
        for ex in s["final_mismatch_examples"][:1]:
            print(f"    Mismatch: q{ex['q_idx']} r{ex['rollout_idx']}: snap={ex['snap_len']} vs ctx={ex['ctx_len']}")
            print(f"    Snap tail: ...{ex['snap_tail'][-100:]}")
            print(f"    Ctx  tail: ...{ex['ctx_tail'][-100:]}")
    print(f"  {'>> OVERALL:':<12} {'PASS' if all_pass else 'FAIL'}")

    # --- Check 5: Tier classification unchanged from v4 ---
    print("\n## Check 5: Tier Classification Unchanged from v4 (±5%)")
    all_pass = True
    for ds in DATASETS:
        if ds not in all_stats:
            continue
        s = all_stats[ds]
        v4 = V4_TIERS.get(ds)
        if not v4:
            continue
        v4_t1 = v4["tier1"]
        v5_t1 = s["tier1"]
        diff = abs(v5_t1 - v4_t1)
        pct_diff = 100 * diff / v4_t1 if v4_t1 > 0 else 0
        status = "PASS" if pct_diff <= 5 else f"FAIL (diff={pct_diff:.1f}%)"
        if pct_diff > 5:
            all_pass = False
        print(f"  {ds:<12} {status} — v4 Tier1={v4_t1}, v5 Tier1={v5_t1}, diff={diff} ({pct_diff:.1f}%)")
    print(f"  {'>> OVERALL:':<12} {'PASS' if all_pass else 'FAIL'}")

    # --- Check 6: No-output rate ---
    print("\n## Check 6: No-Output Rate <5%")
    all_pass = True
    for ds in DATASETS:
        if ds not in all_stats:
            continue
        s = all_stats[ds]
        no_out = s["invoke_no_output"]
        total = s["invoke_total"]
        pct = 100 * no_out / total if total else 0
        status = "PASS" if pct < 5 else f"FAIL ({pct:.1f}%)"
        if pct >= 5:
            all_pass = False
        print(f"  {ds:<12} {status} — {no_out}/{total} invokes with no output ({pct:.1f}%)")
    print(f"  {'>> OVERALL:':<12} {'PASS' if all_pass else 'FAIL'}")

    # --- Spot-checks ---
    print("\n## Spot-Check Examples (Tier 1, R=1, 2 tool calls)")
    for ds in DATASETS:
        if ds not in all_stats:
            continue
        for ex in all_stats[ds].get("spot_checks", []):
            print(f"\n  --- {ds} q{ex['q_idx']} (R={ex['reward']}) ---")
            print(f"  Q: {ex['question'][:120]}")
            for seg in ex["segments"]:
                marker = "HAS [TOOL OUTPUT]" if seg["has_tool_output_marker"] else "clean"
                print(f"  [{seg['type']:>12}] len={seg['snapshot_len']:>6} {marker}")
                # Show first 300 chars of snapshot
                preview = seg["snapshot_preview"][:300]
                for pline in preview.split("\n")[:4]:
                    print(f"    | {pline[:100]}")


if __name__ == "__main__":
    main()
