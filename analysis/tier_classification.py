"""Tier 1/2 classification + diverse rollout generation for critic warmup.

Combines Step 2.4 (Tier 1/2 splits) and Step 3.1 (critic warmup data) in one GPU pass.

For each question:
  Phase 1: 4 no-tool rollouts → Tier 1/2 classification
  Phase 2: 4 tool rollouts (temp=0.7) → segment states for critic
  Phase 3: 4 more tool rollouts (temp=1.0) if Phase 2 had no success (exploration)
  Phase 4: 2 tool rollouts for Tier 2 questions (unnecessary tool use examples)

Usage:
    python -m analysis.tier_classification \
        --input-dir data_local/processed \
        --output-dir data_local/processed \
        --max-samples 5000
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

from vllm import LLM, SamplingParams

from core.code_block_detector import detect_code_block
from core.replacement import replace_code_block
from core.reward import compute_reward, extract_answer
from sandbox.executor import execute_code, extract_search_query_strings

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_TOOL_CALLS = 7
MAX_TOKENS_PER_SEGMENT = 1024

SEARCH_DATASETS = {"hotpotqa", "nq", "musique", "2wiki", "triviaqa"}

SYSTEM_PROMPT_NO_TOOLS = """\
You are a helpful assistant that solves problems step by step.

When you have the final answer, write ONLY the answer inside an answer block:
<answer>42</answer>
<answer>William Shakespeare</answer>
"""

SYSTEM_PROMPT_WITH_TOOLS = """\
You are a helpful assistant that solves problems step by step.

## Tools
You have access to a Python interpreter. Write code in a fenced block:

```python
your code here
```

The code runs in a subprocess — only printed output is captured. Rules:
- You MUST use print() to see any result. Bare expressions like `x` produce no output.
- Variables persist across code blocks in the same conversation.
- Use code ONLY when you need to compute or look up something you don't know.
- Do NOT write code for things you already know.

## <context> block
After EVERY tool output, you MUST immediately write a <context> block extracting the key fact. Keep it short — one or two sentences maximum. Example:

[TOOL OUTPUT]
The Oberoi Group is a hotel group with its head office in Delhi...
<context>Oberoi Group head office: Delhi</context>

Do NOT skip this step. The <context> block is required after every tool output.

## <answer> block
When you have the final answer, write it inside an answer block with no extra words:
<answer>42</answer>
<answer>William Shakespeare</answer>
<answer>Paris</answer>
"""

SEARCH_PROMPT_ADDITION = """
## search() function
A search() function retrieves Wikipedia passages. Always print the result:
```python
results = search("your query here")
print(results)
```
search() returns a formatted string — do NOT index into it with [0]['key']. Just print it and read the output.
"""

log = logging.getLogger(__name__)


def setup_logging(output_dir: Path):
    log_file = output_dir / "tier_classification.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )


def load_model():
    log.info("Loading %s with vLLM...", MODEL_ID)
    t0 = time.time()
    llm = LLM(model=MODEL_ID, trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    log.info("Loaded in %.1fs", time.time() - t0)
    return llm, tokenizer


def load_search_fn():
    try:
        from retrieval.search import get_search
        fn = get_search()
        log.info("Search backend loaded")
        return fn
    except (ImportError, RuntimeError) as e:
        log.warning("No search backend: %s", e)
        return None


def build_prompt(tokenizer, context: str, with_tools: bool, search_enabled: bool) -> str:
    if with_tools:
        system = SYSTEM_PROMPT_WITH_TOOLS + (SEARCH_PROMPT_ADDITION if search_enabled else "")
    else:
        system = SYSTEM_PROMPT_NO_TOOLS
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": context},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def run_no_tool_wavefront(llm, tokenizer, examples: list[dict],
                          num_rollouts: int, params: SamplingParams) -> dict:
    """Run no-tool rollouts for all examples. Returns {question_idx: [rollout_results]}."""
    states = {}
    for ex_idx, ex in enumerate(examples):
        for r_idx in range(num_rollouts):
            key = (ex_idx, r_idx)
            states[key] = {
                "prompt": build_prompt(tokenizer, ex["question"], with_tools=False, search_enabled=False),
                "done": False,
                "text": "",
            }

    # Single generation pass (no tool loop needed)
    keys = list(states.keys())
    prompts = [states[k]["prompt"] for k in keys]

    log.info("No-tool wavefront: %d rollouts", len(prompts))
    outputs = llm.generate(prompts, params)

    results = {}
    for batch_idx, key in enumerate(keys):
        ex_idx, r_idx = key
        text = outputs[batch_idx].outputs[0].text
        ex = examples[ex_idx]
        reward = compute_reward(text, ex["gold_answer"], ex["dataset"])
        pred = extract_answer(text, ex["dataset"])

        if ex_idx not in results:
            results[ex_idx] = []
        results[ex_idx].append({
            "rollout_idx": r_idx,
            "text": text,
            "prediction": pred,
            "reward": reward,
        })

    return results


def run_tool_wavefront(llm, tokenizer, examples: list[dict], indices: list[int],
                       num_rollouts: int, temp: float,
                       search_fn=None) -> dict:
    """Run tool rollouts for selected examples. Returns {question_idx: [rollout_results]}.

    Uses stop strings to enforce real token-by-token execution:
    - invoke wave: stop at closing ``` fence (model can't hallucinate tool output)
    - assimilate wave: stop at </context> (captures real distillation)
    Context accumulates across the full conversation — never reset between waves.
    """
    invoke_params = SamplingParams(
        temperature=temp, top_p=0.9, max_tokens=MAX_TOKENS_PER_SEGMENT,
        stop=["```\n", "```\r\n"],
        include_stop_str_in_output=True,
    )
    assimilate_params = SamplingParams(
        temperature=temp, top_p=0.9, max_tokens=256,
        stop=["</context>"],
        include_stop_str_in_output=True,
    )

    states = {}
    for ex_idx in indices:
        ex = examples[ex_idx]
        search_enabled = ex["dataset"] in SEARCH_DATASETS
        for r_idx in range(num_rollouts):
            key = (ex_idx, r_idx)
            states[key] = {
                "question": ex["question"],
                "context": ex["question"],   # accumulates full conversation
                "search_enabled": search_enabled,
                "segments": [],
                "tool_outputs": [],
                "full_generated": "",
                "done": False,
                "tool_call_count": 0,
                "pending_tool_stdout": None,  # set after invoke, triggers assimilate wave
                "invoke_context_start": None,  # marks where invoke additions begin for phase-2 replacement
            }

    MAX_WAVES = MAX_TOOL_CALLS * 2 + 1
    for wave in range(MAX_WAVES):
        active_keys = [k for k, s in states.items() if not s["done"]]
        if not active_keys:
            break

        log.info("Tool wave %d: %d active", wave, len(active_keys))

        # Split into assimilate vs invoke batches — different stop strings
        assimilate_keys = [k for k in active_keys if states[k]["pending_tool_stdout"] is not None]
        invoke_keys     = [k for k in active_keys if states[k]["pending_tool_stdout"] is None]

        outputs_by_key = {}
        if assimilate_keys:
            prompts = [build_prompt(tokenizer, states[k]["context"],
                                    with_tools=True, search_enabled=states[k]["search_enabled"])
                       for k in assimilate_keys]
            for key, out in zip(assimilate_keys, llm.generate(prompts, assimilate_params)):
                outputs_by_key[key] = out.outputs[0].text
        if invoke_keys:
            prompts = [build_prompt(tokenizer, states[k]["context"],
                                    with_tools=True, search_enabled=states[k]["search_enabled"])
                       for k in invoke_keys]
            for key, out in zip(invoke_keys, llm.generate(prompts, invoke_params)):
                outputs_by_key[key] = out.outputs[0].text

        for key in active_keys:
            state = states[key]
            generated = outputs_by_key[key]
            state["full_generated"] += generated

            # ── Assimilate wave ──────────────────────────────────────────────
            if state["pending_tool_stdout"] is not None:
                stdout = state["pending_tool_stdout"]
                state["pending_tool_stdout"] = None

                m = re.search(r"<context>(.*?)</context>", generated, re.DOTALL | re.IGNORECASE)
                if m:
                    context_text = m.group(1).strip()
                    termination = "context_block"
                else:
                    context_text = generated.strip()[:256]
                    termination = "eos"

                state["segments"].append({
                    "type": "assimilate",
                    "termination": termination,
                    "raw_stdout": stdout,
                    "context_text": context_text,
                })
                # Phase 2 replacement: replace everything from invoke (replaced text +
                # [TOOL OUTPUT] + raw stdout) with just the <context> block.
                # invoke_context_start marks where the invoke additions began.
                invoke_start = state.get("invoke_context_start", len(state["context"]))
                state["context"] = (
                    state["context"][:invoke_start]
                    + f"\n<context>{context_text}</context>\n"
                )
                state["invoke_context_start"] = None
                # Save context snapshot AFTER phase-2 replacement
                state["segments"][-1]["context_snapshot"] = state["context"]
                continue

            # ── Invoke / synthesize wave ─────────────────────────────────────
            code_detection = detect_code_block(generated)

            if code_detection is None:
                state["segments"].append({"type": "synthesize"})
                state["context"] = state["context"] + generated
                state["segments"][-1]["context_snapshot"] = state["context"]
                state["done"] = True
                continue

            search_results = None
            if search_fn and state["search_enabled"]:
                queries = extract_search_query_strings(code_detection.executable)
                if queries:
                    search_results = {q: search_fn(q) for q in queries}

            stdout = execute_code(code_detection.executable,
                                  search_enabled=state["search_enabled"],
                                  search_results=search_results)
            replaced = replace_code_block(generated, code_detection, stdout)

            state["tool_outputs"].append(stdout)
            state["segments"].append({
                "type": "invoke",
                "code": code_detection.executable,
                "output": stdout,
            })
            # Phase 1: append replaced text + tool output banner
            # Track where invoke additions start so phase 2 can replace them
            state["invoke_context_start"] = len(state["context"])
            state["context"] = state["context"] + replaced + f"\n[TOOL OUTPUT]\n{stdout}\n"
            # Save context snapshot AFTER update — this is what the model sees before assimilate
            state["segments"][-1]["context_snapshot"] = state["context"]
            state["full_generated"] += f"\n[TOOL OUTPUT]\n{stdout}\n"
            state["tool_call_count"] += 1
            state["pending_tool_stdout"] = stdout

            if state["tool_call_count"] >= MAX_TOOL_CALLS:
                state["segments"][-1]["termination"] = "truncated"
                state["done"] = True

    # Collect results
    results = {}
    for key, state in states.items():
        ex_idx, r_idx = key
        ex = examples[ex_idx]
        reward = compute_reward(state["context"], ex["gold_answer"], ex["dataset"])
        pred = extract_answer(state["context"], ex["dataset"])

        if ex_idx not in results:
            results[ex_idx] = []
        results[ex_idx].append({
            "rollout_idx": r_idx,
            "text": state["full_generated"],
            "context": state["context"],
            "prediction": pred,
            "reward": reward,
            "num_tool_calls": state["tool_call_count"],
            "segments": state["segments"],
        })

    return results


def process_dataset(llm, tokenizer, data_path: Path, output_dir: Path,
                    search_fn, max_samples: int = 5000):
    examples = [json.loads(line) for line in open(data_path, encoding="utf-8")]
    if max_samples and len(examples) > max_samples:
        # Diverse sample: take evenly spaced indices
        step = len(examples) / max_samples
        indices = [int(i * step) for i in range(max_samples)]
        examples = [examples[i] for i in indices]

    dataset = examples[0]["dataset"]
    log.info("Dataset: %s | N: %d", dataset, len(examples))

    params_normal = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=MAX_TOKENS_PER_SEGMENT)
    params_explore = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=MAX_TOKENS_PER_SEGMENT)
    t0 = time.time()

    # Phase 1: No-tool rollouts (all questions)
    log.info("Phase 1: No-tool rollouts (4 per question)")
    no_tool_results = run_no_tool_wavefront(llm, tokenizer, examples, 4, params_normal)

    # Classify Tier 1/2
    tier1_indices = []
    tier2_indices = []
    for ex_idx in range(len(examples)):
        rollouts = no_tool_results.get(ex_idx, [])
        any_correct = any(r["reward"] > 0 for r in rollouts)
        if any_correct:
            tier2_indices.append(ex_idx)
        else:
            tier1_indices.append(ex_idx)

    log.info("Tier 1 (needs tools): %d | Tier 2 (solvable without): %d",
             len(tier1_indices), len(tier2_indices))

    # Phase 2: Tool rollouts for ALL questions (4 per question, normal temp)
    log.info("Phase 2: Tool rollouts (4 per question, temp=0.7)")
    all_indices = list(range(len(examples)))
    tool_results_normal = run_tool_wavefront(llm, tokenizer, examples, all_indices,
                                             4, temp=0.7, search_fn=search_fn)

    # Phase 3: Extra exploration for questions with no tool success (temp=1.0)
    no_success_indices = []
    for ex_idx in all_indices:
        rollouts = tool_results_normal.get(ex_idx, [])
        if not any(r["reward"] > 0 for r in rollouts):
            no_success_indices.append(ex_idx)

    tool_results_explore = {}
    if no_success_indices:
        log.info("Phase 3: Exploration rollouts for %d questions with no success (temp=1.0)",
                 len(no_success_indices))
        tool_results_explore = run_tool_wavefront(llm, tokenizer, examples,
                                                  no_success_indices, 4, temp=1.0, search_fn=search_fn)

    # Assemble output
    output = []
    for ex_idx, ex in enumerate(examples):
        no_tool = no_tool_results.get(ex_idx, [])
        tool_normal = tool_results_normal.get(ex_idx, [])
        tool_explore = tool_results_explore.get(ex_idx, [])
        all_tool = tool_normal + tool_explore

        any_no_tool_correct = any(r["reward"] > 0 for r in no_tool)
        any_tool_correct = any(r["reward"] > 0 for r in all_tool)
        avg_tool_calls = (sum(r["num_tool_calls"] for r in all_tool) / len(all_tool)) if all_tool else 0

        record = {
            "question_idx": ex_idx,
            "question": ex["question"],
            "gold_answer": ex["gold_answer"],
            "dataset": dataset,
            "tier": "tier2" if any_no_tool_correct else "tier1",
            "any_no_tool_correct": any_no_tool_correct,
            "any_tool_correct": any_tool_correct,
            "avg_tool_calls": round(avg_tool_calls, 2),
            "no_tool_rollouts": no_tool,
            "tool_rollouts": all_tool,
        }
        output.append(record)

    # Write output
    out_path = output_dir / f"{dataset}_tier_splits.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in output:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    elapsed = time.time() - t0
    tier1 = sum(1 for r in output if r["tier"] == "tier1")
    tier2 = sum(1 for r in output if r["tier"] == "tier2")
    tool_success = sum(1 for r in output if r["any_tool_correct"])

    log.info("=== %s Summary ===", dataset)
    log.info("Tier 1 (needs tools): %d/%d (%.1f%%)", tier1, len(output), 100 * tier1 / len(output))
    log.info("Tier 2 (solvable): %d/%d (%.1f%%)", tier2, len(output), 100 * tier2 / len(output))
    log.info("Tool success rate: %d/%d (%.1f%%)", tool_success, len(output), 100 * tool_success / len(output))
    log.info("Total rollouts: %d no-tool + %d tool = %d",
             sum(len(r["no_tool_rollouts"]) for r in output),
             sum(len(r["tool_rollouts"]) for r in output),
             sum(len(r["no_tool_rollouts"]) + len(r["tool_rollouts"]) for r in output))
    log.info("Time: %.1fs (%.2fs per example)", elapsed, elapsed / len(output))
    log.info("Saved to %s", out_path)

    # Also write Tier 1/2 split files (just questions, for training data loading)
    for tier, tier_indices in [("tier1", [i for i, r in enumerate(output) if r["tier"] == "tier1"]),
                                ("tier2", [i for i, r in enumerate(output) if r["tier"] == "tier2"])]:
        split_path = output_dir / f"{dataset}_{tier}.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for idx in tier_indices:
                ex = examples[idx]
                f.write(json.dumps({
                    "question": ex["question"],
                    "gold_answer": ex["gold_answer"],
                    "dataset": dataset,
                    "split": tier,
                }, ensure_ascii=False) + "\n")
        log.info("Wrote %d examples to %s", len(tier_indices), split_path)


def main():
    parser = argparse.ArgumentParser(description="Tier 1/2 classification + diverse rollout generation")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    log.info("Args: %s", vars(args))

    input_dir = Path(args.input_dir)
    # Look for training split files — prefer largest file per dataset
    train_files = sorted(input_dir.glob("*_train_*.jsonl"))
    if not train_files:
        train_files = sorted(input_dir.glob("*_*.jsonl"))
        train_files = [f for f in train_files if "_results" not in f.stem
                       and "_rollout_stats" not in f.stem
                       and "_tier" not in f.stem]

    if args.datasets:
        train_files = [f for f in train_files if any(d in f.stem for d in args.datasets)]

    # Keep only the largest file per dataset (e.g., gsm8k_train_5000 over gsm8k_train_50)
    best_per_dataset = {}
    for f in train_files:
        # Extract dataset name (everything before _train_)
        dataset_name = f.stem.split("_train_")[0] if "_train_" in f.stem else f.stem
        if dataset_name not in best_per_dataset or f.stat().st_size > best_per_dataset[dataset_name].stat().st_size:
            best_per_dataset[dataset_name] = f
    train_files = sorted(best_per_dataset.values())

    if not train_files:
        log.error("No data files found in %s", input_dir)
        return

    log.info("Found %d files: %s", len(train_files), [f.name for f in train_files])

    llm, tokenizer = load_model()
    search_fn = load_search_fn()

    for data_path in train_files:
        log.info("=" * 60)
        log.info("Processing: %s", data_path.name)
        log.info("=" * 60)
        process_dataset(llm, tokenizer, data_path, output_dir,
                        search_fn=search_fn, max_samples=args.max_samples)

    log.info("All done.")


if __name__ == "__main__":
    main()
