"""Standard CoT+Tools baseline (ReAct-style).

A reference implementation of chain-of-thought augmented with tool calls,
following Yao et al. (2022) ReAct. It does not use CARL's segment machinery,
<context> blocks, or two-phase replacement.

    Thought -> Action (tool call) -> Observation -> ... -> Answer

Training-free: relies entirely on the base model's in-context tool-following
ability.
"""
from __future__ import annotations
from carl.core.code_block_detector import detect_code_block
from carl.core.reward import compute_reward, extract_answer


# Per-dataset hints for the system prompt.

_PREAMBLE = (
    "You are a helpful assistant. Solve the problem step by step using the "
    "ReAct pattern: Thought -> Action -> Observation -> ... -> Answer. "
    "Decompose the question, plan your approach, then execute."
)

_TOOL_GRAMMAR = """## Tool

You have a Python interpreter. To call it, write:

Thought: <your reasoning about what to look up or compute>
```python
print(some_expression_or_search("query"))
```

The runtime executes the code and returns its stdout as:

Observation: <stdout from your code>

Rules:
- You MUST use print() to see any result.
- You may iterate Thought -> Action -> Observation as many times as needed.
- When you have the answer, stop calling tools and write:

Answer: <answer>your final answer</answer>
"""

_SEARCH_HINT = """
The function `search("query")` returns Wikipedia passages as a string. Always
print it: `print(search("..."))`.
"""

_TABLE_HINT = """
The question above includes a table inline. You may read it directly or
write Python to compute over the numbers; no retrieval is needed.
"""

_MATH_HINT = """
For arithmetic, use Python rather than mental math. Retrieval is not needed.
"""

_MULTIHOP_HINT = """
This question may need 2-4 tool calls (multi-hop). After each Observation,
plan the next sub-question.
"""

_DATASET_HINT = {
    "gsm8k":    _MATH_HINT,
    "finqa":    _TABLE_HINT,
    "hotpotqa": _SEARCH_HINT,
    "2wiki":    _SEARCH_HINT + _MULTIHOP_HINT,
    "musique":  _SEARCH_HINT + _MULTIHOP_HINT,
}


def _build_system_prompt(mode: str, dataset: str) -> str:
    """mode: 'always' (forced tool use) or 'optional' (model discretion)."""
    if mode == "always":
        usage = ("\nYou MUST use the tool at least once before answering. "
                 "Do not skip it, even on questions you think you can answer directly.")
    else:
        usage = ("\nUse the tool only when helpful or when you are unsure of a fact. "
                 "If you are confident, you may answer directly.")
    return f"{_PREAMBLE}{usage}\n\n{_TOOL_GRAMMAR}\n{_DATASET_HINT.get(dataset, _SEARCH_HINT)}"


def react_rollout(question: str, generate, execute, *, system_prompt: str,
                  dataset: str = "",
                  max_turns: int = 8, max_new_tokens: int = 768) -> dict:
    """One ReAct trajectory.

    `generate(prompt, stop, max_new_tokens)` -> {text, ids, log_probs}
    `execute(code: str) -> str`  (stdout, may be 'ERROR: ...').

    Returns: {full_context, n_tool_calls, n_turns, prediction}.
    """
    context = f"{system_prompt}\n\nQuestion: {question}\n\n"
    n_calls, n_turns = 0, 0

    for _ in range(max_turns):
        n_turns += 1
        out = generate(prompt=context, stop=["```\n", "```\r\n"],
                       max_new_tokens=max_new_tokens)
        text = out["text"]
        context += text

        det = detect_code_block(text + "\n```")
        if det is not None and "```python" in text:
            stdout = execute(det.executable)
            n_calls += 1
            context += "\n```\nObservation: " + stdout + "\n\n"
            continue

        break

    pred = extract_answer(context, dataset)
    return {
        "full_context": context,
        "n_tool_calls": n_calls,
        "n_turns": n_turns,
        "prediction": pred,
    }


def run_cot_tools(prompts, generate, execute, mode: str = "optional", **kw):
    """Drive standard CoT+Tools (ReAct) on a list of evaluation prompts."""
    out = []
    for ex in prompts:
        sys_prompt = _build_system_prompt(mode, ex["dataset"])
        traj = react_rollout(ex["question"], generate, execute,
                             system_prompt=sys_prompt, dataset=ex["dataset"], **kw)
        traj["reward"] = compute_reward(traj["full_context"], ex["gold"], ex["dataset"])
        traj["dataset"] = ex["dataset"]
        traj["q_idx"] = ex["idx"]
        out.append(traj)
    return out
