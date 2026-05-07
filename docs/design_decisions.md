# ToolSMDP — Design Decisions

Locked-in architectural decisions and gotchas. See `paper_draft/toolsmdp/main.tex` for full rationale.

---

## Critic Architecture
- **2-layer MLP head** (hidden dim 1024, ~7.5M params) on shared backbone. Critic gradients flow through backbone.
- **Monte Carlo targets**: V_target = R for all states in episode. Unbiased, bounded variance (binary R, short episodes).
- **3 PPO epochs**, advantages computed once.
- **KL = 0 initially**, add 1e-4 only if EM drops >10% for 50+ steps.

## Segment Mechanics (invoke / assimilate / synthesize)
- **Three segment types**, each an SMDP option:
  - **Invoke**: model generates reasoning + code block. Terminates at closing ``` fence. Code executes, stdout replaces code.
  - **Assimilate**: model reads raw tool output, writes `<context>...</context>` block. Terminates at `</context>`. Raw output replaced by `<context>` block contents.
  - **Synthesize**: free reasoning or final answer. Terminates at EOS or length limit.
- **`<context>` block is learned behavior**, not a forced prompt. System prompt instructs: "After seeing tool output, always write the key result in a `<context>...</context>` block before continuing." Enforced by RL: good assimilations -> correct answers -> positive advantage.
- **Always use `<context>`**, for both search output (distill 500 tokens) and math output (wrap `42` as `<context>347 * 28 = 9716</context>`). Consistent format across all tools.
- **Max 15 segments**. Each tool call = 2 segments (invoke + assimilate). Typical 2-tool trajectory: invoke, assimilate, invoke, assimilate, synthesize = 5 segments.
- **Assimilation budget**: max 256 tokens per `<context>` block.
- **Two-phase replacement**: (1) code -> stdout after invoke, (2) raw stdout -> `<context>` block after assimilate. Both code and raw output are transient.
- **Sequential rollout generation** per question, **batched PPO update** across all segments.
- **GRPO variant excluded** from implementation. Focus purely on PPO.

## Tool Interface
- **Search**: Pyserini BM25 over Wikipedia (21M passages). `get_search()` returns a callable.
  Pre-resolved calls: rollout loop extracts queries from code, resolves in parent process, injects results.
- **Raw Python only** — no separate calculator tool. `search()` is a pre-built function declared in the system prompt. The Python interpreter IS the universal tool.
- **Executor auto-display** — Jupyter-style `_auto_display()` wraps bare last expressions in `print()`. Model writes code naturally; executor handles display.
- **Error replacement**: comments + `"ERROR: message"` preserved in context. Enables recovery learning (Case E in the paper).

## Data & Curriculum
- **Per-batch Tier enforcement**: every batch of 128 has ~90 Tier 1 (needs tools) / ~38 Tier 2 (solvable directly).
- **Linear curriculum ramp** over first 30% of training: start 80% single-tool, ramp to natural distribution.
- **Mixing**: 30/70 NQ/HotpotQA for search; GSM8K for math; 50/25/25 FinQA/GSM8K-subset/HotpotQA-subset for multi-tool.

## Key Decisions Summary
- MATH dataset dropped entirely
- No quantization — always run full precision (float16 on T4, bfloat16 on A100+)
- Pyserini BM25 over Wikipedia (21M passages). One backend, no fallback chain.
- Pre-resolved search calls (rollout loop resolves queries before sandboxed execution)
- Musique HF path: `bdsaglam/musique` (not `drt/musique`)
- TriviaQA HF path: `mandarjoshi/trivia_qa` config `rc`
- VinePPO bookmarked for Milestone 5 — segment-boundary branching for PPO training

---

## Gotchas

1. **Model never sees segment boundaries as markers.** The model generates continuously. `Trajectory`/`Segment` objects are training-loop bookkeeping ONLY. The rollout loop detects boundaries (``` fence, `</context>`, EOS) and splits segments — the model just writes text.

2. **Two-phase replacement erases both code AND raw output.** After the full invoke->assimilate cycle, the context contains only the `<context>` block — neither the code that produced the search nor the full passage returned by the search. The critic evaluates post-assimilation states.

3. **`<context>` detection is a segment boundary.** When the rollout loop sees `</context>`, it: (a) marks the assimilate segment complete, (b) replaces raw tool output with `<context>` contents, (c) starts the next segment (synthesize or invoke). This is analogous to how ``` fence detection works for invoke segments.

4. **Critic warmup is essential.** Without it, initial V(s) predictions are random, making early advantages meaningless noise.

5. **K/V cache must be preserved.** The `<context>` block is part of the model's continuous generation — no prompt injection, no prefix modification. This is why we use a learned `<context>` tag instead of a forced assimilation prompt.
