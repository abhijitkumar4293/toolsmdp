# ToolSMDP — Session Log

Lightweight log of sessions with significant changes (bugs, design decisions, new milestones).
Routine work is tracked in `plan.md` (status updates) and `results.md` (numbers).

---

### Session 1 (2026-03-13): Design Review
No code written. Read both documents, identified 13 spec ambiguities, resolved all with rationale.

### Session 2 (2026-03-13): Milestone 0-1 Implementation
Built all core components and data pipeline:

- **Project setup**: `pyproject.toml`, package inits, container files
- **Container environment**: `Dockerfile`, `docker-compose.yml`, `.devcontainer/devcontainer.json`

### Session 3 (2026-03-13): Container TODOs + Code Cleanup
Resolved all current-milestone container TODOs and simplified code:

- **`.dockerignore`**: Created — excludes `.git`, `__pycache__`, `data_local/`, `checkpoints_local/`, `.devcontainer/`, `*.md` (except `CLAUDE.md`), `.claude/`
- **Dockerfile**: Cleaned up resolved TODOs. Added `sandbox` user + `SANDBOX_USER` env var.
- **`docker-compose.yml`**: Added `stdin_open`/`tty` to `dev`, `--timeout=30` to `test`, new `download` service with HF cache volume.
- **`sandbox/executor.py`**: Simplified — removed `_get_sandbox_uid()` indirection, removed `search_fn` parameter (unused placeholder), removed `_check_imports`/`_build_runner_script` helper split, consolidated into flat `execute_code()` function. 163->87 lines.
- **`data/download_and_format.py`**: Removed trivial one-liner extract functions (hotpotqa, musique, 2wiki -> shared `_answer_field` lambda, finqa -> `_finqa_field` lambda). Removed per-config `output_splits` dict in favor of shared `_SPLIT_DIRS` mapping. Removed empty `metadata: {}` from output. 275->167 lines.
- **`core/code_block_detector.py`**: Post-hoc `detect_code_block()` + real-time `CodeBlockWatcher` state machine. 17 tests in `tests/test_code_block_detector.py`.
- **`sandbox/executor.py`**: Subprocess execution, 5s timeout, import whitelist, `search()` injection. 21 tests in `tests/test_executor.py`.
- **`core/replacement.py`**: Comment-preserving replacement — code vanishes, comments + stdout remain. 10 tests in `tests/test_replacement.py`.
- **`core/reward.py`**: Per-dataset answer extraction (GSM8K `####`, MATH `\boxed{}`, QA patterns, FinQA numeric) + normalized exact match. 30 tests in `tests/test_reward.py`.
- **`core/segment_rollout.py`**: `Segment` and `Trajectory` dataclasses (training-loop bookkeeping only — model never sees these).
- **`data/download_and_format.py`**: Downloads 8 datasets from HuggingFace, converts to unified JSONL. Not yet run (needs HF access on compute platform).

### Session 4 (2026-03-16): Container Validation + Bug Fixes
- **Container validated**: Docker build + `docker compose run test` working on dev machine.
- **100/100 tests passing** after fixing 5 bugs found during first run:
  - `CodeBlockWatcher` EOS detection: check buffer instead of just current token (char-by-char feeding)
  - `executor.py` import guard: switched from allowlist to blocklist — stdlib internal deps (`codecs`, `decoder`, `_io`) were blocked
  - `reward.py` `_extract_last_number` regex: `\.?\d*` -> `(?:\.\d+)?` to avoid matching trailing dots (`42.`)
  - FinQA test expectation: `-3.2` is correct, not `3.2`
- **pyproject.toml**: added `[tool.setuptools.packages.find]` to fix editable install
- **Dockerfile**: split dep install (non-editable) from source install (editable) for layer caching
- **VS Code Dev Container**: working — "Reopen in Container" connects to Linux container with full debugging
- **`.vscode/launch.json`**: added "Debug Tests" and "Debug Current Test File" configurations
- **Sample data downloaded** (50 examples each): `gsm8k_train_50`, `gsm8k_test_50`, `hotpotqa_dev_50`, `finqa_train_50`, `finqa_test_50`

### Session 5 (2026-03-17): Inference Script + Compute Planning
- **`scripts/test_base_model.py`**: Three-mode inference script for Qwen3.5-4B
- **Model choice**: Qwen2.5-3B-Instruct (`Qwen/Qwen2.5-3B-Instruct`), ~6GB VRAM BF16, 128K context
- **Compute plan**: Lightning.ai T4 (79h free) for Milestones 2-3, save A100/H200 for training

### Session 6 (2026-03-21): Search Index + Milestone Planning
- **Milestone tracker**: Created `plan.md` (originally `milestones.md`) — canonical step-by-step tracker
- **Decisions**: MATH dataset dropped, no quantization ever, in-process BM25 (not microservice)
- **Retrieval package** (`retrieval/`): Pyserini BM25 over Wikipedia (21M passages)
- **115/115 tests passing** (15 new search tests)

### Session 7 (2026-03-22): Pyserini Wikipedia Search + Simplification
- Simplified search.py: One backend, one function. `get_search()` returns a callable.
- **Codebase conventions established**: "one way to do things", "flat over nested"
- **110 passed, 2 skipped**

### Session 8 (2026-03-22): Three-Segment Design (invoke/assimilate/synthesize)
- **Paper updated** with three segment types: invoke, assimilate, synthesize
- **`<context>` block design**: learned behavior (system prompt), not forced prompt — preserves K/V cache
- **Max 15 segments**, two-phase replacement, three independent skills trained

### Session 9 (2026-03-23): `<context>` Block Implementation
- `core/context_block_detector.py`, updated `replacement.py`, `segment_rollout.py`
- **138 passed, 2 skipped**

### Session 10 (2026-03-23): GPU Smoke Test + Simplification
- **Model switch**: Qwen3.5-4B -> **Qwen2.5-3B-Instruct** (Qwen3.5 broke vllm)
- **Baseline results**: GSM8K 40% EM, HotpotQA 8% EM (50 samples each)
- **144 passed, 2 skipped**

### Session 11 (2026-03-24): Azure ML Setup
- AML workspace `bingdmml` with Singularity MI300X AMD GPUs
- Custom `amd-inference` Docker environment with Wikipedia BM25 index baked in

### Session 12 (2026-04-08/09): Rollout Loop Fix + Tier v3 Data Generation

**Three critical bugs in rollout loop** invalidated all prior data (v1/v2):
1. **No stop strings** — model hallucinated tool outputs in one shot
2. **Context reset each wave** — prior tool outputs discarded
3. **No assimilate wave** — 0% assimilate segments

Fixes applied. Tier v3 run (`ashy_leather_jjlmj3j646`): 161,660 rollouts, 527,750 segment-boundary states. All 6 critic training scenarios confirmed present.

### Session 13 (2026-04-09): Auto-Display Fix + Tier v4 Re-run

**Bug:** `executor.py` ran code as `.py` subprocess — bare expressions produced no stdout.
79% of GSM8K/FinQA invokes returned "Code produced no output".

**Fix:** `_auto_display()` — AST-based Jupyter-style auto-display. 97.7% fix rate. 158/158 tests.

**Tier v3 invalidated.** Tier v4 re-run submitted: `joyful_basket_n7cw84cr7c`.

**VinePPO bookmarked for Milestone 5:** segment-boundary branching for PPO, not critic warmup.

### Session 14 (2026-04-10): Tier v4 Validation + Milestone 3 Start

**Tier v4 completed** (`joyful_basket_n7cw84cr7c`). Downloaded 1.9GB, 160,712 rollouts, 513,794 segments.

**Step 2.4.1 validation gate — all checks passed:**
- GSM8K no-output: 4.2% (was 79% in v3), FinQA: 0.8%
- I/A ratio: 1.00 across all datasets
- Contrastive: 10,356 questions with mixed R=1/R=0
- Tier shifts minimal (GSM8K -12, others ≤+27)
- Search errors: 5-8% (all <15%)

**Key v4 finding:** Tool success rates dropped from v3's inflated numbers (GSM8K 93%→43%) because
v3's "success" was the model computing in its head during assimilation. Now with real tool output,
failures are genuine wrong computations.

**Milestone 3 unblocked.** Starting Step 3.1: extract critic training pairs.

### Session 15 (2026-04-10): Phase-2 Replacement Bug Found — Tier v4 Invalidated

**Bug:** `tier_classification.py` only implements phase 1 of two-phase replacement.
After assimilation, the loop just appends (`context += generated`). It never replaces
the raw tool output with the `<context>` block. The design spec says:
> "Two-phase replacement: (1) code → stdout after invoke, (2) raw stdout → `<context>`
> block after assimilate. Both code and raw output are transient."

**How found:** When inspecting the actual `context` field of a 2Wiki rollout (Q8, Rollout 7),
the context was 14,732 chars with search results appearing twice. After proper phase-2
replacement, it should be ~200 chars (question + two `<context>` blocks).

**Also found:** The 674K critic warmup pairs extracted in Steps 3.1-3.3 had two problems:
1. Extraction script reconstructed context wrong (put raw code blocks back in,
   used only `context_text` instead of full assimilate generation)
2. Even correct reconstruction would be wrong because the source data itself
   doesn't implement phase-2 replacement

**Additionally:** Segments don't store `context_snapshot` — no way to get intermediate
context states from the stored data.

**Impact:**
- Tier v4 tier classification (Tier 1/2 splits): STILL VALID
- Tier v4 rollout contexts: WRONG
- 674K critic warmup pairs: DISCARDED
- Milestone 3: RE-BLOCKED

**Fix needed:**
1. Implement phase-2 replacement in `tier_classification.py`
2. Save `context_snapshot` in every segment dict
3. Re-run as tier v5
4. Re-extract critic training pairs from v5

### Session 16 (2026-04-10): Tier v5 Submission Fix

**`clever_street_plhcztx9md` failed** — missing `--input-dir` and `--output-dir` CLI args.
The `aml submit` command didn't include the arguments that `tier_classification.py` requires.

**Re-submitted as `honest_tooth_wr13z6lhky`** with correct command:
```
python -m analysis.tier_classification --input-dir data_local/processed --output-dir ./outputs --max-samples 5000
```

### Session 17 (2026-04-11): Tier v5 Validation + Milestone 2 Closeout

**Tier v5 completed and validated.** `honest_tooth_wr13z6lhky` — 156,916 tool rollouts, 736,568 segments.

**All 6 automated validation checks passed:**
- Phase-2 replacement: 0/289,826 assimilate snapshots have bare `[TOOL OUTPUT]` (PASS)
- Context shrinks after assimilate: 89-99.5% (PASS)
- `context_snapshot` in every segment: 736,568/736,568 (PASS)
- Final snapshot matches rollout context: 156,916/156,916 (PASS)
- Tier classification unchanged from v4: all within ±1.3% (PASS)
- No-output rate: GSM8K 5.1%, others <1% (PASS)

**Key finding:** Some `<context>` blocks contain `[TOOL OUTPUT]` text (model parroting search results
in its distillation) — but this is model behavior, not a code bug. Phase-2 replacement works correctly.

**v5 vs v4 tool success jumped:** GSM8K 43%→95%, HotpotQA 43%→58%, 2Wiki 28%→56%.
Clean context from phase-2 replacement helps the model perform much better.

**Critic training data extracted:** 582,165 warmup pairs (invoke + assimilate boundaries only),
5,185 easy + 11,567 hard calibration anchors, 10,830 contrastive questions.

**Extraction script rewritten:** `extract_critic_pairs.py` now uses `segment["context_snapshot"]`
directly instead of reconstructing from segment data.

**Milestone 2: DONE.** Milestone 3 unblocked — next step is Step 3.4 (train critic head).

### Session 18 (2026-04-12): Critic Training Data Analysis + Step 3.4 Launch

**Deep data characterization** of 582K critic warmup pairs across 5 dimensions:
- Tool call count: R=1 concentrates at 2 calls (sweet spot), R=0 spreads flatter
- Invoke outcomes: 83-98% code OK, GSM8K/FinQA have 15-18% errors
- Hollow R=1 (all invokes error, still correct): GSM8K 7.1%, FinQA 14.4%, others <1%
- Assimilate quality: 93-99.7% clean distillation, 5-7% parroting on search datasets

**Story D analysis (FinQA/Musique):** 93-95% V=0. Root cause: FinQA 60% wrong computation,
Musique 88% search found nothing. Decided to keep both datasets — signal exists but weak.

**Tier 2 tools-hurt analysis:** Tools reduce per-rollout success by 20-40pp on easy questions.
Root causes: 54% bad distillation (GSM8K), 51% bad search (HotpotQA), 20% lost track (4+ calls).
This is correct signal — critic should learn tools hurt on easy questions (selectivity).

**Step 3.4 launched:** `affable_camel_vm8x6sly09` (critic-warmup-v1).
Script: `training/train_critic.py`. Architecture: Qwen2.5-3B + 2-layer MLP head.
3 epochs, 524K train pairs, MSE + calibration loss. Expected ~4-6h.

### Session 19 (2026-04-13): Critic Training Debugging — SDPA Backward Bug on ROCm

**Multiple failed attempts** to train the critic on MI300X (AMD ROCm):

1. **v1-v3**: dtype mismatch (`float32` targets vs `bfloat16` head output). Fixed with `.float()` upcast in ValueHead.
2. **v4-v5**: OOM with `model.to(device)` (single GPU die ~48GB). Fixed with `device_map="auto"`.
3. **v5**: NaN at step 1 in both bf16 and fp32. Initially suspected precision, then device_map + gradient checkpointing interaction.
4. **v6**: Added `torch.autograd.set_detect_anomaly(True)` — **found the real bug**:
   `ScaledDotProductEfficientAttentionBackward0` returns NaN on AMD ROCm.
   This is a known bug in AMD's fused attention backward kernel. Forward works, backward doesn't.

**Fix:** `attn_implementation="eager"` — forces standard PyTorch math attention (separate matmul + softmax ops instead of fused kernel). Reliable backward, ~2× slower but correct.

**Memory solution:** Eager attention materializes N×N attention matrices (~270GB for all layers).
`gradient_checkpointing` reduces this to 1 layer at a time (~7.5GB). Combined with `device_map="auto"`, batch 64 fits.

**Frozen backbone test** (`calm_spider`): Head-only training failed — val expl_var=-0.049 after full epoch. Pretrained hidden representations don't contain enough value-prediction signal. Backbone must be unfrozen.

**Smoke test results** (5K samples, 1 epoch, fp32, eager, grad checkpoint):

| Batch | Val loss | Expl var | v1_mean | v0_mean | Separation | Time |
|-------|----------|----------|---------|---------|------------|------|
| 16 | 0.237 | 0.047 | 0.510 | 0.452 | 0.058 | 35 min |
| **64** | **0.236** | **0.064** | **0.624** | **0.512** | **0.112** | **15 min** |
| Frozen | 0.270 | -0.049 | 0.461 | 0.434 | 0.027 | 9 min |

**Full run launched:** `plum_planet_ymd38jjbkj` (critic-FINAL-b64).
Config: 582K samples, 1 epoch, batch 64, fp32, eager, grad checkpoint, early stopping (patience 5).
Expected: ~30h (likely ~15-20h with early stopping).

**Other changes:**
- `--attn`, `--grad-checkpoint`, `--freeze-backbone`, `--debug`, `--grad-accum` CLI flags added
- Early stopping with best checkpoint saving
- Val subsampled to 2000 for speed
- `CLAUDE.md`: added "no wandb" convention
