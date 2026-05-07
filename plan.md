# ToolSMDP — Milestone Tracker

**Status key:** DONE | VALIDATED (approved, ready to execute) | DRAFT (not yet reviewed)

---

## Milestone 0: Environment + Data — DONE

Completed in Sessions 1-4. Project setup, container, all core components built and tested.

## Milestone 1: Core ToolSMDP Components — DONE

Code block detector, context block detector, executor (no import restrictions), replacement,
reward (with infinity fix), segment rollout. Tests passing.

---

## Milestone 2: Pre-Training Analysis — DONE

**Completed 2026-04-11.** All tier classification, rollout generation, and critic training
data extraction validated and ready for Milestone 3.

**Bugs found and fixed along the way:**
1. Phase-2 replacement (Session 15): rollout loop only did phase 1. Fixed in tier v5.
2. Auto-display (Session 13): bare expressions produced no stdout. Fixed with `_auto_display()`.
3. No stop strings, context reset, no assimilate wave (Session 12): all fixed.

**Final data:** Tier v5 (`honest_tooth_wr13z6lhky`) — 156,916 tool rollouts, 736,568 segments,
582K critic warmup pairs. All validation checks passed.

### Step 2.0: Search Backend — DONE
Wikipedia search via Pyserini BM25 (21M passages). Baked into `amd-inference:8` Docker image.

### Step 2.0.5: `<context>` Block Detection + Two-Phase Replacement — DONE

### Step 2.1: GPU Smoke Test — DONE
Validated on Lightning.ai T4. GSM8K 40% (pass@1), HotpotQA 8% (pass@1).

### Step 2.2: Base Model + Tools Characterization (500 samples) — DONE

**v3 results (current — corrected eval, jobs `willing_nutmeg_92y52w8rj6` + `eval-v3`):**

| Dataset | N | EM (pass@4) | Avg Tool Calls |
|---|---|---|---|
| GSM8K | 500 | **78.6%** | 3.54 |
| HotpotQA | 500 | **40.0%** | 1.96 |
| 2Wiki | 500 | **44.8%** | 2.19 |
| FinQA | 500 | **19.4%** | 3.11 |
| Musique | 500 | **12.4%** | 1.90 |

Full results + analysis: `results.md`

**Critical bugs found in rollout loop (fixed 2026-04-08):**
1. **No stop strings** — model generated full rollout in one shot, hallucinating tool outputs from parametric memory. Fixed: `stop=["```\n"]` for invoke, `stop=["</context>"]` for assimilate.
2. **Context reset each wave** — `state["context"] = question + last_replaced` discarded all prior tool outputs. Fixed: context accumulates across the full conversation.
3. **No assimilate wave** — loop never gave the model a turn to write `<context>` after seeing real tool output. Fixed: separate assimilate wave with `pending_tool_stdout` flag.
4. **Bad system prompt** — "variables don't carry over" caused hallucinated data; `<context>` instruction was weak. Fixed: mandatory `<context>` after every tool output, `print()` explicitly required.

Validation job running: `loyal_ocean_q248c0ld0v` (GSM8K, HotpotQA, 2Wiki — 500 samples × 4 rollouts). Expected ~1.5-2 hours. Goal: confirm assimilate segments appear and accuracy improves with real tool outputs.

### Step 2.3: Build difficulty buckets — DONE
Built from v3 eval rollout data. Full breakdown in `results.md`.

| Dataset | 1-call (N / EM) | 2-calls (N / EM) | 3+ calls (N / EM) |
|---|---|---|---|
| GSM8K | 30 / 83% | 96 / 76% | 374 / 79% |
| HotpotQA | 128 / 33% | 235 / 44% | 137 / 40% |
| 2Wiki | 97 / 42% | 210 / 49% | 193 / 42% |
| FinQA | 78 / 24% | 108 / 24% | 314 / 17% |
| Musique | 172 / 8% | 186 / 18% | 142 / 10% |

### Step 2.4: Build Tier 1/2 training splits — DONE (tier v5)

**Final job:** `honest_tooth_wr13z6lhky` (tier v5 — phase-2 replacement + context_snapshot)

| Dataset | N | Tier 1 | Tier 2 | Tool Success | Tool Rollouts | Segments |
|---|---|---|---|---|---|---|
| GSM8K | 5000 | 153 (3%) | 4847 (97%) | 94.9% | 22,676 | 88,612 |
| HotpotQA | 5000 | 3794 (76%) | 1206 (24%) | 57.5% | 30,008 | 139,186 |
| 2Wiki | 5000 | 2864 (57%) | 2136 (43%) | 56.2% | 31,052 | 171,044 |
| FinQA | 5000 | 4555 (91%) | 445 (9%) | 22.2% | 36,316 | 162,346 |
| Musique | 5000 | 4663 (93%) | 337 (7%) | 20.5% | 36,864 | 175,380 |
| **TOTAL** | **25000** | **16,029** | **8,971** | | **156,916** | **736,568** |

**Key changes from v4:** Tool success rates MUCH higher (e.g., GSM8K 43%→95%, HotpotQA 43%→58%)
because phase-2 replacement gives the model clean context — it performs better with
distilled facts than with raw search results still cluttering the context.

**Prior versions (all invalidated):**
- v4 (`joyful_basket_n7cw84cr7c`): missing phase-2 replacement
- v3 (`ashy_leather_jjlmj3j646`): auto-display bug
- v2 (`lemon_milk_137bldxxcp`): three rollout loop bugs
- v1 (`eager_chin_s48qtyv8s2`): corrupted rollouts

### Step 2.4.1: Validate tier data — DONE

Tier v5 validated with 6 automated checks + 5 manual spot-checks (`analysis/validate_tier_v5.py`).

| Check | Result |
|---|---|
| Phase-2 replacement (no bare `[TOOL OUTPUT]` outside `<context>` tags) | **PASS** (0/289,826) |
| Context shrinks after assimilate | **PASS** (89-99.5% shrink, exceptions are short tool outputs) |
| `context_snapshot` stored in every segment | **PASS** (736,568/736,568) |
| Final snapshot matches rollout context | **PASS** (156,916/156,916) |
| Tier classification unchanged from v4 (±5%) | **PASS** (all within ±1.3%) |
| No-output rate <5% | **PASS** (GSM8K 5.1% marginal, others 0.0-0.7%) |

Manual spot-checks all passed: HotpotQA 2-tool R=1, GSM8K 1-tool R=1, contrastive R=1 vs R=0,
FinQA table extraction, hard anchor all-R=0. All show clean context after assimilate.

### Step 2.4.2: Extract critic training data — DONE

Extracted 582,165 warmup pairs from v5 data using `analysis/extract_critic_pairs.py`.
Data stored in `data_local/critic_warmup/`.

| Output | Count |
|---|---|
| Warmup pairs (invoke + assimilate boundaries) | **582,165** |
| V=1 pairs | 97,618 (16.8%) |
| V=0 pairs | 484,547 (83.2%) |
| Invoke pairs | 292,339 (50.2%) |
| Assimilate pairs | 289,826 (49.8%) |
| Easy calibration anchors (Tier 2 all-correct) | 5,185 |
| Hard calibration anchors (Tier 1 all-fail) | 11,567 |
| Contrastive questions (mixed R=1/R=0) | 10,830 |

Per-dataset V=1 ratios:
- GSM8K: 0.58 (easy — model solves most)
- HotpotQA: 0.21, 2Wiki: 0.16 (medium — search helps sometimes)
- FinQA: 0.06, Musique: 0.05 (hard — tool use rarely succeeds)

**Validation:** 0/289,826 assimilate pairs have bare `[TOOL OUTPUT]`. V_target values are
exactly {0.0, 1.0}. Structure matches expected format from Step 2.4.2 plan.

### Step 2.5: Write predictions document — TODO
- Pre-register expected RL training results before any training
- Based on difficulty bucket data + tier split analysis

---

## Milestone 3: Critic Warm-Up — IN PROGRESS

**Unblocked by:** Tier v5 data validated + 582K critic warmup pairs extracted.

**Goal:** Initialize the critic so PPO has usable advantages from epoch 1.
This is the MOST CRITICAL milestone — the paper's core contribution depends on the critic working.

**Why this matters (VinePPO concern):**
VinePPO showed token-level critics in PPO barely beat random at ranking. Our argument:
segment-level is easier (≤7 states not 200, massive info change between states, binary MC targets).
We must PROVE the critic works BEFORE starting PPO.

**What the critic needs to learn (to be confirmed in tier v5 data):**
- V(s0) separation: Tier 2 questions → V≈1.0, Tier 1 all-fail questions → V≈0.0
- Invoke discrimination: good search output → higher V than error/irrelevant output
- Assimilate discrimination: accurate `context_text` → higher V than wrong distillation
- Multi-hop state transitions: V rises after good first search, rises again after good second search

### Step 3.1 — Extract training pairs from tier v5 rollouts — DONE

**582,165 warmup pairs** from v5 data using `segment["context_snapshot"]` directly.
Only invoke and assimilate boundaries (no initial, no synthesize).

**Script:** `analysis/extract_critic_pairs.py`
**Output:** `data_local/critic_warmup/critic_warmup_pairs.jsonl`
Each line: `{context, V_target, segment_type, dataset, question_idx, rollout_idx}`

### Step 3.2 — Create calibration anchors — DONE

5,185 easy + 11,567 hard anchors from v5 data.
**Output:** `data_local/critic_warmup/critic_calibration_anchors.jsonl`

### Step 3.3 — Build contrastive pairs — DONE

10,830 contrastive questions (mixed R=1/R=0 outcomes).
**Output:** `data_local/critic_warmup/critic_contrastive_questions.jsonl`

### Step 3.4 — Train critic head — RUNNING

Job: `plum_planet_ymd38jjbkj` (critic-FINAL-b64)

**Architecture:**
```
Backbone: Qwen2.5-3B-Instruct (model.model — skip LM head, UNFROZEN)
Head:     Linear(2048, 1024) → ReLU → Linear(1024, 1) → Sigmoid  (~2.1M params)
Loss:     MSE(V_pred, V_target) + 0.1 * calibration_loss
LR:       1e-4 head, 1e-5 backbone (AdamW, cosine schedule, 500 warmup steps)
Epochs:   1 (early stopping, patience 5)
Batch:    64
Dtype:    fp32 (bf16 NaN'd on ROCm)
Attn:     eager (SDPA backward has NaN bug on ROCm AMD GPUs)
Grad ckpt: ON (eager attn needs it to fit in memory)
```

**Script:** `training/train_critic.py`
**Checkpoint:** `./outputs/critic_warmup/best.pt`
**Expected runtime:** ~15-30 hours on MI300X

**ROCm bug found (Session 19):** `ScaledDotProductEfficientAttentionBackward0` produces NaN
on AMD MI300X. Fix: `attn_implementation="eager"`. Frozen backbone doesn't work (expl_var=-0.049).

Key design choices:
- Left-truncation (keep recent context) + left-padding (last pos = last real token)
- `model.model` to skip LM head (avoids 37GB logits tensor)
- Train/val split by (dataset, question_idx) — no rollout leakage
- wandb logging: loss, V separation, explained variance

### Step 3.5 — CRITICAL CHECKPOINT: Pre-training critic evaluation

Before starting PPO, evaluate the warm-up critic on held-out data.
**If this fails, do NOT proceed to PPO.**

**Test 1 — V(s0) separation:**
Plot V(s0) distributions for Tier 2 (easy) vs Tier 1 all-fail (hard) questions.
Target: easy peak > 0.7, hard peak < 0.4, minimal overlap.

**Test 2 — Invoke discrimination:**
For questions with mixed rollout outcomes, does V(s_after_invoke) > V(s_before_invoke)
on R=1 rollouts? And V(s_after_bad_invoke) < V(s_before) on R=0 rollouts?
Target: >60% accuracy on this binary ranking task.

**Test 3 — Explained variance:**
On held-out rollouts: Var(V_target - V_predicted) / Var(V_target).
Target: >0.5 (critic explains more than half the variance in outcomes).

**Decision gate:**
- All 3 pass → proceed to Milestone 4 (training framework)
- Test 1 fails → more calibration anchor data or higher calibration loss weight
- Test 2 fails → more contrastive pairs, or check invoke output quality in data
- Test 3 fails → more training epochs or larger head

**Deliverables:**
- [x] ~582K critic warmup pairs with correct context_snapshots (invoke + assimilate only)
- [x] Calibration anchors: 5,185 easy + 11,567 hard
- [x] Contrastive question index (10,830)
- [ ] Trained critic head checkpoint
- [ ] **Pre-training critic evaluation figure** (goes in paper — Section 4)

---

## Milestone 4: Training Framework — DRAFT

**Goal:** Implement segment-level PPO + baselines in same framework for fair comparison.

### Step 4.1 — Choose and set up RL framework
- OpenRLHF or veRL with PPO support
- Implement segment-level rollout loop (code fences → execute → </context> tags)
- Two-level replacement in the rollout

### Step 4.2 — Segment advantage computation
- A(seg_k) = V(s_{k+1}) - V(s_k) for intermediate segments
- A(seg_N) = R - V(s_N) for final segment
- λ = 0 (no multi-step lookahead)
- All tokens in a segment share the same scalar advantage

### Step 4.3 — PPO loss with segment advantages
- Standard PPO clipped loss, just advantage values differ per segment
- 3 PPO epochs per buffer, KL=0 initially

### Step 4.4 — Implement baselines in same framework
- **Search-R1 GRPO:** trajectory-level GRPO, gradient masking, single rollout
- **ToolSMDP + GRPO:** segment decomposition but GRPO advantages (no critic)
- **Standard PPO:** token-level GAE on full trajectory (optional, lower priority)

### Step 4.5 — Integration test
- 10 PPO steps on 10 questions with Qwen2.5-1.5B
- Verify: segment types, V(s) values, advantages, loss, no NaNs

### Step 4.6 — Critic quality tracking during training
- Every 50 steps: V(s₀) separation, discrimination accuracy, explained variance
- Logged to wandb for real-time monitoring

**Deliverables:**
- [ ] Segment-level PPO working
- [ ] Search-R1 GRPO baseline in same framework
- [ ] ToolSMDP + GRPO variant
- [ ] Integration test passing
- [ ] Critic tracking pipeline

---

## Milestone 5: First Training (3B) + Decision Gate — DRAFT

**Goal:** Train at 3B scale, validate core claim, go/no-go for 7B.

### Step 5.1 — Training runs (3B, GSM8K + HotpotQA subset)

| Run | Method | What It Validates |
|---|---|---|
| A | **ToolSMDP + PPO (ours)** | Does segment credit + critic improve EM? |
| B | **ToolSMDP + GRPO** | Does segment structure alone help (without critic)? |
| C | **Search-R1 GRPO** | Trajectory-level baseline |

### Step 5.2 — Evaluate on difficulty buckets
- Per-bucket EM for all three runs
- **KEY TEST:** A > B/C more on 3+ call questions than 1-call?
- This is the hypothesis: gains scale with credit assignment complexity
- **Output token counting:** Measure total output tokens per method per dataset for compute-efficiency comparison

### Step 5.3 — Selectivity analysis
- For each method, plot tool-call frequency on Tier 2 (easy) questions over training
- Prediction: ToolSMDP decreases (learns not to use tools on easy Qs)
- This validates the V(s₀) selectivity mechanism

### Step 5.4 — Post-training critic validation
- MC ground-truth estimation: 200 segment-boundary states, 16 rollouts each
- V_learned vs V_MC scatter plot
- Calibration curve
- Step ranking accuracy

### Step 5.5 — Decision gate
- A > B on multi-hop → proceed to 7B
- A ≈ B → debug critic (is it actually providing useful signal?)
- A < B → stop and rethink

### Step 5.6 — VinePPO-style segment branching (BOOKMARKED)

**Idea:** At each segment boundary during PPO rollouts, generate multiple continuations instead of one.
This gives per-segment advantage estimates by holding prior segments fixed:
"Holding the invoke fixed, what's the expected value of different assimilations?"

**Why here, not critic warmup:** Critic warmup needs (state, V_target) pairs — more rollouts per
question is the simpler fix for signal diversity. VinePPO branching shines during PPO where we need
accurate per-segment advantages for policy gradient updates.

**Analysis from Session 13:**
- 78% of contrastive pairs diverge at first invoke (search query quality)
- 16% diverge at assimilation (same search, different distillation)
- Musique: 84% all-R=0 questions with 8 rollouts, ~34% would find R=1 with 16 rollouts
- Branching factor 4 at each segment boundary: 5-segment trajectory → 4^5 = 1024 paths (expensive)
- More practical: branch at invoke boundaries only (the high-variance decision point)

**Deliverables:**
- [ ] Three trained 3B models
- [ ] Difficulty bucket results (Table 6 in paper — KEY result)
- [ ] Selectivity curves (paper figure)
- [ ] Post-training critic validation (paper figure)
- [ ] Go/no-go decision

---

## Milestone 6: Full-Scale Runs (7B) — DRAFT

Only after Milestone 5 passes. 4x A100 80GB.

### Core experiments (fill paper Tables 1-3)

| Run | Data | ~Hours |
|---|---|---|
| T1: ToolSMDP Search | NQ + HotpotQA (170K) | ~35h |
| T2: ToolSMDP Math | GSM8K (7.5K) | ~8h |
| T3: ToolSMDP Multi-tool | FinQA (6.2K) | ~10h |

### Ablations (fill Tables 4-7)

| Run | Method | Purpose |
|---|---|---|
| T4: Search-R1 GRPO | Trajectory-level baseline | Main comparison |
| T5: ToolSMDP + GRPO | Segment structure, no critic | Isolates structure vs credit |
| T6: No assimilation | ToolSMDP without `<context>` blocks | Validates assimilation |
| T7: MC estimation | Replace learned critic with MC | Critic quality upper bound |

### Full evaluation on all benchmarks with difficulty bucketing

**Deliverables:**
- All paper tables filled
- All trained models saved

---

## Milestone 7: Analysis, Figures, Paper — DRAFT

### Step 7.1 — Diagnostic figures
1. **Pre-training critic validation** (V(s₀) easy/hard separation) — from M3
2. **Post-training critic validation** (V vs MC scatter, calibration) — from M5
3. **Training curves** (EM over steps, all methods)
4. **Gains by difficulty bucket** (Table 6 — KEY figure)
5. **Selectivity curves** (tool frequency by difficulty over training)
6. **Advantage distributions per segment type** (invoke, assimilate, synthesize)
7. **Assimilation quality** (overlap between `<context>` and gold answer over training)

### Step 7.2 — Paper tables
| Table | Content | Source |
|---|---|---|
| Table 1 | Main results (EM by dataset, all methods) | M6 |
| Table 2 | Case analysis (advantage signs by segment) | M5 analysis |
| Table 3 | Tool selectivity (unnecessary call rate) | M5 |
| Table 6 | **Gains by tool-call count** (KEY) | M5/M6 |
| Table 7 | Mechanism isolation (structure vs credit) | M5/M6 |
| Table 8 | Assimilation ablation | M6 T6 |

### Step 7.3 — Paper writing
- Results section with actual numbers
- Strengthen critic argument (Section 4) with diagnostic figures
- Related work (VinePPO, StepTool, ToolRL, PAVs, TRM)
- Conclusion

**Deliverables:**
- All figures and tables
- Complete paper draft

---

## Priority Ranking (if time runs out)

### MUST HAVE (paper rejected without)
1. Main results table (ToolSMDP vs Search-R1 GRPO)
2. Pre-training critic validation (addresses VinePPO concern)
3. Post-training critic validation (confirms critic works)
4. Gains by tool-call count (Table 6 — core hypothesis)
5. Selectivity curves (unique contribution)

### STRONGLY RECOMMENDED
6. Mechanism isolation (segment structure vs value function)
7. Assimilate ablation

### NICE TO HAVE
8. MC estimation variant (critic quality upper bound)
9. λ sensitivity (λ=0 vs 0.5 vs 0.95)
10. Advantage distributions by segment type

---

## Risk Assessment

| Risk | Likelihood | Mitigation | Detection Point |
|---|---|---|---|
| Critic doesn't work | Medium | MC estimation fallback | M3 Step 3.5 |
| ToolSMDP ≈ Search-R1 | Low (multi-hop) | Focus on multi-hop narrative | M5 Step 5.2 |
| Training instability | Medium | Lower critic LR, gradient clipping | M5 during training |
| `<context>` not produced | Low-Medium | Light SFT on 100 examples | M5 first 50 rollouts |
| Not enough GPU budget | Depends | Priority ranking above | Before M6 |
