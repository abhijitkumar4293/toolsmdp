"""Segment-level rollout loop (paper Figure 1, Section 3.1, Appendix B).

The loop drives one trajectory by repeatedly calling
`generate(prompt, stop, max_new_tokens)`, detects code blocks (invoke
segments) and <context> blocks (assimilate segments), runs phase-1
replacement (code body -> stdout) after invoke and phase-2 replacement
(raw stdout -> <context> block) after assimilate.

`generate(prompt, stop, max_new_tokens) -> {text, ids, log_probs}` is
backed by either HF `model.generate` (eval) or vLLM continuous batching
(training); see `carl.eval.hf_generator` and `carl.ppo.trainer`.
"""
from __future__ import annotations
from typing import Callable

from carl.core.segment import Segment, Trajectory
from carl.core.code_block_detector import detect_code_block
from carl.core.context_block_detector import detect_context_block
from carl.core.replacement import replace_tool_output_with_context

GenFn = Callable[..., dict]
ExecFn = Callable[[str], str]


CARL_SYSTEM_PROMPT = """You are a helpful assistant that answers questions, optionally using Python.

To use a tool, write a Python code block:

```python
# describe what you are checking
result = search("query string")
print(result)
```

The runtime executes the block and appends its stdout. You MUST then write a `<context>` block distilling ONLY the facts you need:

<context>key fact 1; key fact 2</context>

You may iterate (search -> context) up to a few times, then commit your answer:

<answer>your final answer</answer>
"""


def rollout(
    question: str,
    generate: GenFn,
    execute: ExecFn,
    max_segments: int = 15,
    invoke_max_tokens: int = 512,
    assimilate_max_tokens: int = 256,
    synthesize_max_tokens: int = 512,
    system_prompt: str = CARL_SYSTEM_PROMPT,
) -> Trajectory:
    context = f"{system_prompt}\n\nQuestion: {question}\n\n"
    traj = Trajectory()
    pending_stdout_block: str | None = None

    for _ in range(max_segments):
        if pending_stdout_block is not None:
            seg_prompt = context
            out = generate(prompt=seg_prompt, stop=["</context>"],
                           max_new_tokens=assimilate_max_tokens)
            gen_text = out["text"]
            full_gen = gen_text if gen_text.endswith("</context>") else gen_text + "</context>"

            ctx_det = detect_context_block(full_gen)
            if ctx_det is not None:
                context = replace_tool_output_with_context(
                    seg_prompt, pending_stdout_block, full_gen,
                )
                term = "context_block"
            else:
                context = seg_prompt + full_gen
                term = "truncated"

            traj.segments.append(Segment(
                start_context=seg_prompt,
                generated_text=full_gen,
                generated_ids=out.get("ids", []),
                log_probs=out.get("log_probs", []),
                segment_type="assimilate", termination=term,
            ))
            pending_stdout_block = None
            continue

        seg_prompt = context
        out = generate(prompt=seg_prompt, stop=["```\n", "```\r\n"],
                       max_new_tokens=invoke_max_tokens)
        gen_text = out["text"]
        code_det = detect_code_block(gen_text + "\n```")

        if code_det is not None and "```python" in gen_text:
            stdout = execute(code_det.executable)
            stdout_block = "[TOOL OUTPUT]\n" + stdout + "\n"
            full_gen = gen_text + "\n```"
            traj.segments.append(Segment(
                start_context=seg_prompt,
                generated_text=full_gen,
                generated_ids=out.get("ids", []),
                log_probs=out.get("log_probs", []),
                segment_type="invoke", termination="tool_call",
                tool_code=code_det.executable,
                tool_comments=code_det.comments,
                tool_output=stdout,
            ))
            context = seg_prompt + full_gen + "\n" + stdout_block
            pending_stdout_block = stdout_block
        else:
            traj.segments.append(Segment(
                start_context=seg_prompt,
                generated_text=gen_text,
                generated_ids=out.get("ids", []),
                log_probs=out.get("log_probs", []),
                segment_type="synthesize", termination="eos",
            ))
            context = seg_prompt + gen_text
            break

    # Reaching max_segments without a synthesize commit means the last
    # segment did not terminate cleanly.
    if traj.segments and traj.segments[-1].segment_type != "synthesize":
        if traj.segments[-1].termination not in ("truncated",):
            traj.segments[-1].termination = "truncated"

    traj.full_context = context
    return traj
