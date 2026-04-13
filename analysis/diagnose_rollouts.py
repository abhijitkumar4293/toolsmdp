"""
Deep diagnosis of rollout failures:
1. What search code patterns cause errors?
2. Does the model ever produce <context> blocks?
3. What does a "working" vs "broken" search rollout look like?
"""
import json
from pathlib import Path
from collections import Counter

TIER_DIR = Path("downloads/tier_v2/artifacts/outputs")
EVAL_DIR = Path("downloads/eval_v3/artifacts/outputs")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ── 1. Search error taxonomy ──────────────────────────────────────────────────
print("=" * 70)
print("1. SEARCH ERROR TAXONOMY (HotpotQA Tier1, first 500 questions)")
print("=" * 70)

rows = load_jsonl(TIER_DIR / "hotpotqa_tier_splits.jsonl")
t1 = [r for r in rows if r["tier"] == "tier1"][:500]

error_types = Counter()
error_examples = {}

for r in t1:
    for ro in r.get("tool_rollouts", []):
        for seg in ro.get("segments", []):
            if seg.get("type") != "invoke":
                continue
            out = seg.get("output", "")
            code = seg.get("code", "")
            if "ERROR" not in out and out.strip():
                continue
            # Classify error
            if "Code produced no output" in out:
                # Figure out why: no print, returns nothing
                if "print(" not in code and "results" not in code.split("=")[0]:
                    key = "no_print_no_assignment"
                elif "results[0][" in code or "results[0]." in code:
                    key = "index_into_string"
                elif "['" in code and "results" in code:
                    key = "dict_access_on_string"
                else:
                    key = "no_output_other"
            elif "NameError" in out:
                key = "NameError"
            elif "TypeError" in out:
                key = "TypeError"
            elif "ImportError" in out or "ModuleNotFoundError" in out:
                key = "bad_import"
            elif "AttributeError" in out:
                key = "AttributeError"
            elif "JSONDecodeError" in out or "requests" in code:
                key = "http_request"
            elif "IndexError" in out:
                key = "IndexError"
            else:
                key = "other_error"

            error_types[key] += 1
            if key not in error_examples:
                error_examples[key] = (code[:300], out[:200])

for k, n in error_types.most_common():
    print(f"  {k:<35} {n:>5}")
    c, o = error_examples[k]
    print(f"    code: {c[:120].strip()!r}")
    print(f"    out:  {o[:100].strip()!r}")
    print()


# ── 2. Context block audit ────────────────────────────────────────────────────
print("=" * 70)
print("2. CONTEXT BLOCK AUDIT — does the model EVER write <context>?")
print("=" * 70)

for ds in ["gsm8k", "hotpotqa", "2wiki", "musique"]:
    rows = load_jsonl(TIER_DIR / f"{ds}_tier_splits.jsonl")
    t1 = [r for r in rows if r["tier"] == "tier1"]

    has_context = 0
    total_rollouts = 0
    context_examples = []

    for r in t1:
        for ro in r.get("tool_rollouts", []):
            total_rollouts += 1
            text = ro.get("text", "") or ro.get("context", "")
            if "<context>" in text.lower():
                has_context += 1
                if len(context_examples) < 2:
                    context_examples.append((r["question"][:100], text[:400]))

    print(f"\n{ds}: {has_context}/{total_rollouts} rollouts have <context> ({has_context/total_rollouts*100:.2f}%)")
    for q, ex in context_examples:
        print(f"  Q: {q}")
        print(f"  Text: {ex[:300]}")


# ── 3. What does a working search rollout look like? ─────────────────────────
print()
print("=" * 70)
print("3. WORKING SEARCH ROLLOUTS (HotpotQA Tier1, reward=1)")
print("=" * 70)

rows = load_jsonl(TIER_DIR / "hotpotqa_tier_splits.jsonl")
t1 = [r for r in rows if r["tier"] == "tier1"]

shown = 0
for r in t1:
    if shown >= 3:
        break
    for ro in r.get("tool_rollouts", []):
        if ro.get("reward", 0) > 0 and shown < 3:
            print(f"\nQ: {r['question'][:200]}")
            print(f"Gold: {r['gold_answer']}")
            print(f"Tool calls: {ro['num_tool_calls']}  Reward: {ro['reward']}")
            print("--- Full text ---")
            print(ro["text"][:1500])
            shown += 1
            break


# ── 4. What does a broken search rollout look like? ──────────────────────────
print()
print("=" * 70)
print("4. BROKEN SEARCH ROLLOUTS (HotpotQA Tier1, reward=0, 3+ calls)")
print("=" * 70)

shown = 0
for r in t1:
    if shown >= 2:
        break
    for ro in r.get("tool_rollouts", []):
        if ro.get("reward", 0) == 0 and ro.get("num_tool_calls", 0) >= 3 and shown < 2:
            print(f"\nQ: {r['question'][:200]}")
            print(f"Gold: {r['gold_answer']}")
            print("--- Full text ---")
            print(ro["text"][:2000])
            shown += 1
            break


# ── 5. System prompt audit ───────────────────────────────────────────────────
print()
print("=" * 70)
print("5. WHAT SYSTEM PROMPT WAS THE MODEL GIVEN?")
print("=" * 70)
# Look for system prompt in the analysis script or pre_training_characterization
import ast, re
src = Path("analysis/pre_training_characterization.py").read_text(encoding="utf-8")
# Find SYSTEM_PROMPT
m = re.search(r'SYSTEM_PROMPT\s*=\s*(""".*?""")', src, re.DOTALL)
if not m:
    m = re.search(r"SYSTEM_PROMPT\s*=\s*('''.*?''')", src, re.DOTALL)
if m:
    print(m.group(1)[:2000])
else:
    # Just grep for it
    for i, line in enumerate(src.splitlines()):
        if "system" in line.lower() and "prompt" in line.lower():
            print(f"  line {i}: {line}")
