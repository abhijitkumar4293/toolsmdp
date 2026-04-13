# ToolSMDP — Project Guide

## What This Project Is

Segment-level RL for LLM tool use via the Semi-Markov Decision Process framework. Trains LLMs to use tools (Python code blocks) with per-segment credit assignment using PPO and a learned value function. Three segment types — **invoke** (call tool), **assimilate** (distill tool output into `<context>` block), **synthesize** (reason/answer) — each an SMDP option with independent advantage. Good tool calls get positive advantage, bad assimilations get independently negative advantage, unnecessary calls get near-zero advantage — all from a single binary outcome reward.

## Key Files

| File | Purpose |
|------|---------|
| `plan.md` | **Canonical plan tracker** — milestones, steps, status, what to do next |
| `results.md` | Experiment results — accuracy tables, job history, feeds into paper |
| `docs/design_decisions.md` | Locked-in architecture decisions + gotchas |
| `docs/sessions.md` | Session history (lightweight — only bugs, design changes, milestones) |
| `paper_draft/toolsmdp/main.tex` | Paper draft — the plan derives from this |

**Skills:** `/aml` (job management), `/plan` (project planning from plan.md + results.md)

## Codebase Conventions

- **This is research code, not production code.** Keep it simple, readable, and small.
- **One way to do things.** Don't build multiple backends, fallback chains, or abstraction layers. Pick one approach and use it directly.
- **Flat over nested.** Functions over classes when a class would just have one method. No inheritance hierarchies. No factory patterns.
- **No unnecessary abstractions.** Three similar lines > a premature helper function. No wrappers around libraries that add no value.
- **Minimal docstrings.** A one-liner is fine. Multi-paragraph docstrings are not. Let function names and signatures speak.
- **No defensive coding** for impossible scenarios. Trust internal code paths.
- Modular package structure: `core/`, `sandbox/`, `data/`, `retrieval/`, `tests/`
- All paths via `$DATA_ROOT` and `$CHECKPOINT_ROOT` env vars (compute-agnostic)
- Container-first: all deps managed via `pyproject.toml`, never `pip install` on bare metal
- **No wandb** — AML environment has no API key configured. All training scripts should default to `wandb_enabled=False`. Log metrics to stdout instead.

## How to Run

```bash
# Run tests (local with conda)
conda run -n toolsmdp pytest tests/ -v

# Run tests (Docker)
docker compose run test

# Download datasets
python -m data.download_and_format --datasets gsm8k hotpotqa finqa --max-samples 50

# Test search locally
python scripts/try_search.py

# Base model inference (requires GPU)
python -m scripts.test_base_model --mode code_gen              # smoke test
python -m scripts.test_base_model --mode pipeline              # full SMDP loop
python -m scripts.test_base_model --mode eval --data data_local/eval_splits/hotpotqa_dev_50.jsonl
```

## Experiment Results Management

**Every experiment run must be documented.** After downloading results from a job:

1. **Store results** in `downloads/<descriptive-name>/` with the job name recorded.
2. **Update `results.md`** with the new numbers (tables, per-dataset, per-bucket).
3. **Update `analysis/step_X_X_analysis.md`** with:
   - Summary of what changed (sandbox fix, data format, etc.)
   - Before/after comparison tables
   - Concrete failure examples (question, gold, prediction, tool calls)
   - Insights on why things improved or didn't
4. **Track result lineage** — record which job, environment version, code changes, and data version produced each set of numbers.

**Result files:**
- `results.md` — living summary of all numbers, feeds into paper tables
- `analysis/step_X_X_analysis.md` — detailed per-step analysis with examples and failure modes
- `downloads/` — raw result artifacts organized by job name
