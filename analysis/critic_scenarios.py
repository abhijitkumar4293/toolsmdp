"""
Pick 2 representative questions per dataset and show the full segment flow
across multiple rollouts, illustrating the contrastive scenarios the critic needs.

For each question we want to show:
- Same question, different rollout outcomes (R=1 vs R=0)
- What differed at the segment level (good invoke vs bad invoke, good assimilate vs bad)
- What V_target each segment-boundary state gets
"""
import json
from pathlib import Path
from collections import defaultdict

TIER_DIR = Path("downloads/tier_v3/artifacts/outputs")
DATASETS = ["gsm8k", "hotpotqa", "2wiki", "finqa", "musique"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def show_rollout(ro, indent="  "):
    segs = ro.get("segments", [])
    r = ro.get("reward", 0)
    pred = ro.get("prediction", "")
    print(f"{indent}Rollout {ro['rollout_idx']} — R={r}  pred={pred!r}  segs={[s['type'] for s in segs]}")
    for i, seg in enumerate(segs):
        t = seg["type"]
        if t == "invoke":
            code_short = seg.get("code", "")[:80].replace("\n", " ").strip()
            out = seg.get("output", "")
            out_short = out[:100].replace("\n", " ").strip()
            has_error = "ERROR" in out
            print(f"{indent}  Seg{i} INVOKE  code={code_short!r}")
            print(f"{indent}         output={'ERROR' if has_error else out_short!r}")
        elif t == "assimilate":
            term = seg.get("termination", "?")
            ctx = seg.get("context_text", "").replace("\n", " ").strip()[:120]
            print(f"{indent}  Seg{i} ASSIMILATE [{term}]  ctx={ctx!r}")
        elif t == "synthesize":
            print(f"{indent}  Seg{i} SYNTHESIZE → answer")


def find_examples(rows, scenario):
    """Find a question matching a given scenario."""
    if scenario == "correct_clean":
        # R=1 rollout with full invoke→assimilate→synthesize and no errors
        for r in rows:
            for ro in r.get("tool_rollouts", []):
                segs = ro.get("segments", [])
                types = [s["type"] for s in segs]
                outputs = [s.get("output","") for s in segs if s["type"]=="invoke"]
                if (ro.get("reward",0) > 0
                        and "assimilate" in types
                        and not any("ERROR" in o for o in outputs)):
                    return r
    elif scenario == "contrastive_same_q":
        # Question where some rollouts R=1, some R=0 — best for critic
        for r in rows:
            tool = r.get("tool_rollouts", [])
            rewards = [ro.get("reward",0) for ro in tool]
            if any(x > 0 for x in rewards) and any(x == 0 for x in rewards):
                return r
    elif scenario == "bad_invoke_good_assimilate":
        # Invoke errored but assimilate still wrote something, R=0
        for r in rows:
            for ro in r.get("tool_rollouts", []):
                segs = ro.get("segments", [])
                has_error_invoke = any(
                    s["type"]=="invoke" and "ERROR" in s.get("output","")
                    for s in segs)
                has_assimilate = any(s["type"]=="assimilate" for s in segs)
                if has_error_invoke and has_assimilate and ro.get("reward",0)==0:
                    return r
    elif scenario == "tier2_used_tools":
        # Tier 2 question (solvable without tools) but model used tools anyway
        for r in rows:
            if r.get("tier") == "tier2":
                tool = r.get("tool_rollouts", [])
                if any(ro.get("num_tool_calls",0) > 0 for ro in tool):
                    return r
    return None


for ds in DATASETS:
    rows = load_jsonl(TIER_DIR / f"{ds}_tier_splits.jsonl")
    tier1 = [r for r in rows if r["tier"] == "tier1"]
    tier2 = [r for r in rows if r["tier"] == "tier2"]

    print(f"\n{'='*70}")
    print(f"DATASET: {ds.upper()}")
    print(f"{'='*70}")

    # ── Question 1: Contrastive same question (best critic training signal) ──
    ex1 = find_examples(rows, "contrastive_same_q")
    if ex1:
        tool = ex1.get("tool_rollouts", [])
        r1_rollouts = [ro for ro in tool if ro.get("reward",0) > 0][:2]
        r0_rollouts = [ro for ro in tool if ro.get("reward",0) == 0][:2]
        print(f"\nQ1 (CONTRASTIVE — same question, mixed outcomes):")
        print(f"  Q: {ex1['question'][:120]}")
        print(f"  Gold: {ex1['gold_answer']}")
        print(f"  Tier: {ex1['tier']} | total rollouts: {len(tool)}")
        print(f"\n  ── R=1 rollouts (critic target: all seg-states → V=1.0) ──")
        for ro in r1_rollouts:
            show_rollout(ro)
        print(f"\n  ── R=0 rollouts (critic target: all seg-states → V=0.0) ──")
        for ro in r0_rollouts:
            show_rollout(ro)
        print(f"\n  CRITIC LEARNING: same s0, different outcomes.")
        print(f"  V(s0) converges to fraction of rollouts with R=1 = {sum(ro.get('reward',0)>0 for ro in tool)}/{len(tool)}")

    # ── Question 2: Tier 2 using tools unnecessarily ──────────────────────────
    ex2 = find_examples(rows, "tier2_used_tools")
    if ex2:
        tool = ex2.get("tool_rollouts", [])
        notool = ex2.get("no_tool_rollouts", [])
        print(f"\nQ2 (TIER 2 — model knew answer, tools unnecessary):")
        print(f"  Q: {ex2['question'][:120]}")
        print(f"  Gold: {ex2['gold_answer']}")
        print(f"  No-tool rollouts: {[ro.get('reward',0) for ro in notool]} (all correct)")
        print(f"\n  ── Tool rollouts (model used tools it didn't need) ──")
        for ro in tool[:3]:
            show_rollout(ro)
        print(f"\n  CRITIC LEARNING: V(s0) should be high (model already knows).")
        print(f"  Invoke segments on Tier2 questions → near-zero advantage (tool not needed).")

print(f"\n{'='*70}")
print("SCENARIO COVERAGE SUMMARY")
print(f"{'='*70}")
scenarios = {
    "Same q, R=1 rollout":       "V(s_k) = 1.0 for all k — critic learns good trajectories",
    "Same q, R=0 rollout":       "V(s_k) = 0.0 for all k — critic learns failure modes",
    "Tier2 with tools":          "V(s0) high, invoke advantage ≈ 0 — selectivity signal",
    "Bad invoke (error)":        "Invoke output=ERROR → low V(s_after_invoke) when R=0",
    "Good invoke, bad assimilate":"Invoke found answer, assimilation lost it → assimilate penalized",
    "Multi-hop (2+ invokes)":    "V rises after good search, falls after bad — per-step signal",
}
for s, desc in scenarios.items():
    print(f"  {s:<35} {desc}")
