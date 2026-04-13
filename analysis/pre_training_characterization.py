"""Pre-training characterization: run base model with tools on eval sets.

Uses vLLM with wavefront batching: all questions × all rollouts are processed
together. Each generation step batches every active rollout into one vLLM call,
maximizing GPU utilization.

Usage:
    python -m analysis.pre_training_characterization \
        --input-dir data_local/eval_splits \
        --output-dir data_local/analysis \
        --num-rollouts 4

    # Quick test
    python -m analysis.pre_training_characterization \
        --input-dir data_local/eval_splits \
        --output-dir data_local/analysis \
        --max-samples 2 --num-rollouts 1
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
from core.reward import compute_reward, extract_answer, normalize_answer
from sandbox.executor import execute_code, extract_search_query_strings

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_TOOL_CALLS = 7
MAX_TOKENS_PER_SEGMENT = 1024

SEARCH_DATASETS = {"hotpotqa", "nq", "musique", "2wiki", "triviaqa"}

SYSTEM_PROMPT = """\
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
After EVERY tool output, you MUST immediately write a <context> block extracting the key fact needed to answer the question. Keep it short — one or two sentences maximum. Example:

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
    log_file = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    log.info("Logging to %s", log_file)


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
        log.info("Search backend loaded (Pyserini)")
        return fn
    except (ImportError, RuntimeError) as e:
        log.warning("No search backend: %s", e)
        return None


def build_prompt(tokenizer, context: str, search_enabled: bool) -> str:
    system = SYSTEM_PROMPT + (SEARCH_PROMPT_ADDITION if search_enabled else "")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": context},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def classify_tool_type(code: str) -> str:
    has_search = "search(" in code
    lines = [l.strip() for l in code.strip().splitlines()
             if l.strip() and not l.strip().startswith("#")]
    non_search_lines = [l for l in lines
                        if "search(" not in l and l not in ("", "print(results)")]
    has_calc = len(non_search_lines) > 0
    if has_search and has_calc:
        return "both"
    return "search" if has_search else "calc"


def compute_tool_output_relevance(tool_output: str, gold_answer) -> float:
    if not tool_output or tool_output.startswith("ERROR:"):
        return 0.0
    gold_str = gold_answer if isinstance(gold_answer, str) else " ".join(gold_answer)
    gold_tokens = set(normalize_answer(gold_str).split())
    if not gold_tokens:
        return 0.0
    output_tokens = set(normalize_answer(tool_output).split())
    return len(gold_tokens & output_tokens) / len(gold_tokens)


# ── Wavefront rollout engine ──

def process_dataset(llm, tokenizer, data_path: Path, output_dir: Path,
                    num_rollouts: int, search_fn, max_samples: int = 0):
    """Process all questions × rollouts using wavefront batching.

    All active rollouts across all questions are batched into each vLLM
    generate call. Rollouts finish independently as they produce final
    answers or exhaust tool calls.
    """
    examples = [json.loads(line) for line in open(data_path)]
    if max_samples:
        examples = examples[:max_samples]

    dataset = examples[0]["dataset"]
    search_enabled = dataset in SEARCH_DATASETS
    log.info("Dataset: %s | N: %d | Rollouts: %d | Wavefront batch size: %d",
             dataset, len(examples), num_rollouts, len(examples) * num_rollouts)

    # invoke: stop at closing code fence so model can't hallucinate tool output
    invoke_params = SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=MAX_TOKENS_PER_SEGMENT,
        stop=["```\n", "```\r\n"],
        include_stop_str_in_output=True,
    )
    # assimilate: stop at </context> so we capture exactly the distillation
    assimilate_params = SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=256,
        stop=["</context>"],
        include_stop_str_in_output=True,
    )

    # Initialize state for every (question, rollout) pair
    # Each state is identified by (example_idx, rollout_idx)
    #
    # context accumulates the full conversation: question + all generated text so far.
    # After each tool call, we append "[TOOL OUTPUT]\n{stdout}\n" and let the model
    # continue — first to write <context>...</context> (assimilate), then to either
    # call another tool (invoke) or give a final answer (synthesize).
    states = {}
    for ex_idx, ex in enumerate(examples):
        for r_idx in range(num_rollouts):
            key = (ex_idx, r_idx)
            states[key] = {
                "question": ex["question"],
                "context": ex["question"],   # grows with every wave
                "segments": [],
                "tool_types": [],
                "tool_outputs": [],
                "full_generated": "",
                "done": False,
                "tool_call_count": 0,
                "pending_tool_stdout": None, # set after invoke, cleared after assimilate
            }

    t0 = time.time()

    # Each wave is one generation step. Rollouts in "assimilate" mode get their
    # context ended with the raw tool output so the model writes <context>.
    # Rollouts in "invoke" mode get the full accumulated context so far.
    MAX_WAVES = MAX_TOOL_CALLS * 2 + 1  # invoke + assimilate per tool call, plus final synthesize
    for wave in range(MAX_WAVES):
        active_keys = [k for k, s in states.items() if not s["done"]]
        if not active_keys:
            break

        log.info("Wave %d: %d active rollouts", wave, len(active_keys))

        # Split active rollouts into assimilate (pending tool output) vs invoke/synthesize
        assimilate_keys = [k for k in active_keys if states[k]["pending_tool_stdout"] is not None]
        invoke_keys = [k for k in active_keys if states[k]["pending_tool_stdout"] is None]

        outputs_by_key = {}
        if assimilate_keys:
            prompts = [build_prompt(tokenizer, states[k]["context"], search_enabled)
                       for k in assimilate_keys]
            for key, out in zip(assimilate_keys, llm.generate(prompts, assimilate_params)):
                outputs_by_key[key] = out.outputs[0].text
        if invoke_keys:
            prompts = [build_prompt(tokenizer, states[k]["context"], search_enabled)
                       for k in invoke_keys]
            for key, out in zip(invoke_keys, llm.generate(prompts, invoke_params)):
                outputs_by_key[key] = out.outputs[0].text

        for key in active_keys:
            state = states[key]
            generated = outputs_by_key[key]
            state["full_generated"] += generated

            # ── Assimilate wave: model should write <context>...</context> ──
            if state["pending_tool_stdout"] is not None:
                stdout = state["pending_tool_stdout"]
                state["pending_tool_stdout"] = None

                # Extract <context> block if present
                m = re.search(r"<context>(.*?)</context>", generated, re.DOTALL | re.IGNORECASE)
                if m:
                    context_text = m.group(1).strip()
                    termination = "context_block"
                else:
                    # Model skipped <context> — treat its text as the distillation
                    context_text = generated.strip()[:256]
                    termination = "eos"

                state["segments"].append({
                    "type": "assimilate",
                    "termination": termination,
                    "raw_stdout": stdout,
                    "context_text": context_text,
                })
                # Append the assimilation text then continue to next wave
                state["context"] = state["context"] + generated
                continue

            # ── Invoke / synthesize wave ──
            code_detection = detect_code_block(generated)

            if code_detection is None:
                # No code block — synthesize, rollout done
                state["segments"].append({"type": "synthesize", "termination": "eos"})
                state["context"] = state["context"] + generated
                state["done"] = True
                continue

            # Execute code
            search_results = None
            if search_fn and search_enabled:
                queries = extract_search_query_strings(code_detection.executable)
                if queries:
                    search_results = {q: search_fn(q) for q in queries}

            stdout = execute_code(
                code_detection.executable,
                search_enabled=search_enabled,
                search_results=search_results,
            )
            replaced = replace_code_block(generated, code_detection, stdout)

            tt = classify_tool_type(code_detection.executable)
            state["tool_types"].append(tt)
            state["tool_outputs"].append(stdout)
            state["segments"].append({
                "type": "invoke", "termination": "tool_call",
                "tool_type": tt, "code": code_detection.executable, "output": stdout,
            })

            # Append generated text (with code replaced by stdout) to context,
            # then append the raw tool output banner so the model sees it and
            # writes <context> in the next (assimilate) wave.
            state["context"] = state["context"] + replaced + f"\n[TOOL OUTPUT]\n{stdout}\n"
            state["full_generated"] += f"\n[TOOL OUTPUT]\n{stdout}\n"
            state["tool_call_count"] += 1
            state["pending_tool_stdout"] = stdout  # triggers assimilate wave next

            if state["tool_call_count"] >= MAX_TOOL_CALLS:
                state["segments"][-1]["termination"] = "truncated"
                state["done"] = True

    # Mark any still-active as done
    for s in states.values():
        s["done"] = True

    # Aggregate results per question
    results = []
    for ex_idx, ex in enumerate(examples):
        rollouts = []
        for r_idx in range(num_rollouts):
            s = states[(ex_idx, r_idx)]
            reward = compute_reward(s["context"], ex["gold_answer"], dataset)
            pred = extract_answer(s["context"], dataset)
            relevances = [compute_tool_output_relevance(out, ex["gold_answer"])
                          for out in s["tool_outputs"]]

            rollouts.append({
                "rollout_idx": r_idx,
                "num_segments": len(s["segments"]),
                "num_tool_calls": sum(1 for seg in s["segments"] if seg["type"] == "invoke"),
                "tool_types": s["tool_types"],
                "tool_output_relevance": relevances,
                "reward": reward,
                "prediction": pred,
                "full_generated": s["full_generated"],
                "final_context": s["context"],
                "segments": s["segments"],
            })

        avg_tool_calls = sum(r["num_tool_calls"] for r in rollouts) / num_rollouts
        any_correct = any(r["reward"] > 0 for r in rollouts)

        if avg_tool_calls < 1.5:
            bucket = "1_call"
        elif avg_tool_calls < 2.5:
            bucket = "2_calls"
        else:
            bucket = "3+_calls"

        result = {
            "question_idx": ex_idx,
            "question": ex["question"],
            "gold_answer": ex["gold_answer"],
            "dataset": dataset,
            "avg_tool_calls": avg_tool_calls,
            "avg_segments": sum(r["num_segments"] for r in rollouts) / num_rollouts,
            "bucket": bucket,
            "any_correct": any_correct,
            "rollouts": rollouts,
        }
        results.append(result)

    # Write results
    out_path = output_dir / f"{dataset}_rollout_stats.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    total = len(results)
    correct = sum(1 for r in results if r["any_correct"])
    elapsed = time.time() - t0
    log.info("=== %s Summary ===", dataset)
    log.info("EM (any_correct): %d/%d = %.1f%%", correct, total, 100 * correct / total)

    from collections import Counter
    buckets = Counter(r["bucket"] for r in results)
    for b in ["1_call", "2_calls", "3+_calls"]:
        n = buckets.get(b, 0)
        if n == 0:
            continue
        bc = sum(1 for r in results if r["bucket"] == b and r["any_correct"])
        log.info("  %s: %d questions, EM = %d/%d = %.1f%%", b, n, bc, n, 100 * bc / n)

    log.info("Avg tool calls: %.2f", sum(r["avg_tool_calls"] for r in results) / total)
    log.info("Total time: %.1fs (%.1fs per example)", elapsed, elapsed / total)
    log.info("Saved %d results to %s", total, out_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="Pre-training characterization (vLLM wavefront)")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    log.info("Args: %s", vars(args))

    input_dir = Path(args.input_dir)
    eval_files = sorted(input_dir.glob("*_*.jsonl"))
    eval_files = [f for f in eval_files if "_results" not in f.stem and "_rollout_stats" not in f.stem]
    if args.datasets:
        eval_files = [f for f in eval_files if any(d in f.stem for d in args.datasets)]
    if not eval_files:
        log.error("No eval files found in %s", input_dir)
        return
    log.info("Found %d eval files: %s", len(eval_files), [f.name for f in eval_files])

    llm, tokenizer = load_model()
    search_fn = load_search_fn()

    for data_path in eval_files:
        log.info("=" * 60)
        log.info("Processing: %s", data_path.name)
        log.info("=" * 60)
        process_dataset(llm, tokenizer, data_path, output_dir,
                        num_rollouts=args.num_rollouts, search_fn=search_fn,
                        max_samples=args.max_samples)

    log.info("All done.")


if __name__ == "__main__":
    main()
