"""Print 5 detailed examples showing token-by-token segment flow."""
import json
from pathlib import Path

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

hotpot = load("downloads/eval_v4/artifacts/outputs/hotpotqa_rollout_stats.jsonl")
wiki   = load("downloads/eval_v4/artifacts/outputs/2wiki_rollout_stats.jsonl")
gsm    = load("downloads/eval_v4/artifacts/outputs/gsm8k_rollout_stats.jsonl")

def show(label, r, ro):
    fg = ro["full_generated"]
    print(f"\n{'='*70}")
    print(f"EXAMPLE: {label}")
    print(f"{'='*70}")
    print(f"Question : {r['question']}")
    print(f"Gold     : {r['gold_answer']}")
    print(f"Predicted: {ro['prediction']}")
    print(f"Reward   : {ro['reward']}  |  Tool calls: {ro['num_tool_calls']}  |  Segments: {ro['num_segments']}")
    print(f"\n--- FULL GENERATION (annotated) ---")

    # Walk through and annotate segment boundaries
    pos = 0
    seg_num = 0
    text = fg

    # Split on [TOOL OUTPUT] markers and <context> tags to show boundaries
    import re
    # We'll just print the raw text but insert visible markers
    # Annotate: code fences, [TOOL OUTPUT], <context>, <answer>
    annotated = text
    annotated = re.sub(r'```python', '\n[SEG:INVOKE starts → model writes code]\n```python', annotated)
    annotated = re.sub(r'```\n\[TOOL OUTPUT\]', '```\n[← closing fence: code execution happens here]\n[TOOL OUTPUT]', annotated)
    annotated = re.sub(r'\[TOOL OUTPUT\]\n', '[TOOL OUTPUT — real stdout injected]\n', annotated)
    annotated = re.sub(r'<context>', '\n[SEG:ASSIMILATE starts → model writes distillation]\n<context>', annotated)
    annotated = re.sub(r'</context>', '</context>\n[← </context>: assimilate segment ends]', annotated)
    annotated = re.sub(r'<answer>', '\n[SEG:SYNTHESIZE → final answer]\n<answer>', annotated)
    print(annotated)
    print(f"\n{'─'*70}")

# ── Example 1: Simple 1-hop search, correct ──────────────────────────────────
for r in hotpot:
    for ro in r["rollouts"]:
        if ro["reward"] > 0 and ro["num_tool_calls"] == 1 and "<context>" in ro.get("full_generated",""):
            show("1 — HotpotQA: simple 1-hop, CORRECT", r, ro)
            break
    else: continue
    break

# ── Example 2: 2-hop search, correct ─────────────────────────────────────────
for r in wiki:
    for ro in r["rollouts"]:
        fg = ro.get("full_generated","")
        if ro["reward"] > 0 and ro["num_tool_calls"] == 2 and fg.count("<context>") >= 2:
            show("2 — 2Wiki: 2-hop search, CORRECT", r, ro)
            break
    else: continue
    break

# ── Example 3: Search returns bad result, wrong answer ───────────────────────
for r in hotpot:
    for ro in r["rollouts"]:
        fg = ro.get("full_generated","")
        if ro["reward"] == 0 and ro["num_tool_calls"] >= 1 and "<context>" in fg and "[TOOL OUTPUT]" in fg:
            # Find one where the stdout looks non-empty but answer is wrong
            if "ERROR" not in fg.split("[TOOL OUTPUT]")[1][:50]:
                show("3 — HotpotQA: search returned output but answer WRONG", r, ro)
                break
    else: continue
    break

# ── Example 4: GSM8K multi-step math ─────────────────────────────────────────
for r in gsm:
    for ro in r["rollouts"]:
        if ro["reward"] > 0 and ro["num_tool_calls"] >= 2:
            show("4 — GSM8K: multi-step math, CORRECT", r, ro)
            break
    else: continue
    break

# ── Example 5: Search errors then recovers with parametric knowledge ──────────
for r in hotpot:
    for ro in r["rollouts"]:
        fg = ro.get("full_generated","")
        parts = fg.split("[TOOL OUTPUT]")
        has_error = any("ERROR" in p[:60] for p in parts[1:])
        if has_error and ro["reward"] > 0:
            show("5 — HotpotQA: search errored, model recovered, CORRECT", r, ro)
            break
    else: continue
    break
