# CARL: Competence-Aware Reinforcement Learning for LLM Tool Use

This package contains: segment decomposition + two-phase replacement, SMDP-derived per-segment advantages, segment-level PPO trainer (OpenRLHF subclass), Search-R1 baseline adapters, the CoT+Tools baseline, the eval harness reproducing Table 1, plus calibration / error / hallucination / resilience analyses for the appendix tables.

```
carl_release/
  carl/                  # the package
    core/                # rollout, segment dataclasses, code/context detectors, replacement, EM
    sandbox/             # subprocess Python sandbox + auto-display
    retrieval/           # BM25 (Pyserini) + MockSearch
    data/                # dataset download, Tier 1/2 labeler
    critic/              # 2-layer MLP value head + warm-up trainer
    ppo/                 # segment-level PPO trainer + Eq. 1 advantages
    baselines/           # CoT+Tools, Search-R1 adapter
    eval/                # full eval, tool-absence resilience, hallucination
    analysis/            # calibration metrics, error categorization
  scripts/               # numbered entry points (01..06)
  configs/               # carl_3b.yaml, carl_7b.yaml, sr1_ppo_3b.yaml, sr1_grpo_3b.yaml
  dockerfiles/           # NVIDIA + AMD images used for training and eval
  sample_data/           # representative artifacts at the two stages tests need
  tests/{unit,smoke}     # unit + smoke tests
```

## Install

```bash
pip install -e .
# Optional extras:
pip install -e .[retrieval]   # Pyserini + faiss for BM25 over Wikipedia
pip install -e .[training]    # accelerate + deepspeed + vLLM + OpenRLHF
```

### Reproducing the training environment

The training and eval runs were produced inside two container images, one
per accelerator. Both Dockerfiles ship in this repo:

| Vendor | Base image | Dockerfile |
|---|---|---|
| AMD MI300X | `rocm/vllm:rocm6.3.1_mi300_ubuntu22.04_py3.12_vllm_0.6.6` | [`dockerfiles/Dockerfile.amd`](dockerfiles/Dockerfile.amd) |
| NVIDIA H100 / A100 | `nvidia/cuda:12.4.0-devel-ubuntu22.04` | [`dockerfiles/Dockerfile.nvidia`](dockerfiles/Dockerfile.nvidia) |

Both images include PyTorch + vLLM + transformers + datasets + Pyserini
(with the Wikipedia DPR-100w BM25 index pre-downloaded into Pyserini's
cache). Build either one with:

```bash
docker build -f dockerfiles/Dockerfile.amd    -t carl-amd    .
# or
docker build -f dockerfiles/Dockerfile.nvidia -t carl-nvidia .
```

Then mount the repo and install the package on top of the image:

```bash
docker run --gpus all -it -v $PWD:/workspace carl-nvidia bash
# inside the container:
pip install -e .[training]
```

These were used on Azure ML Singularity (MI300X cluster, ND24is_MI300X_v5);
they run unchanged on a workstation with Docker + the matching driver.

## End-to-end pipeline

| Phase | Script | Output |
|---|---|---|
| 1. Data prep | `scripts/01_prepare_data.py` | `data_local/processed/{ds}_{split}.jsonl` |
| 2. Tier 1/2 labeling | `scripts/02_tier_classify.py` | `tier_labels.jsonl` |
| 3a. Build warm-up pairs | `scripts/03_extract_warmup.py` | `data_local/critic_warmup/critic_warmup_pairs.jsonl` |
| 3b. Train critic | `scripts/04_train_critic.py` | `outputs/critic_warmup/best.pt` |
| 4. CARL PPO | `scripts/05_train_carl_ppo.py` | `outputs/carl_3b/` (or 7b/) |
| 4'. Baselines | `carl/baselines/search_r1/README.md` | upstream verl checkpoints |
| 5. Full eval | `scripts/06_eval_full.py` | `results/eval/{tag}_{predictions,metrics}.{jsonl,json}` |

### Concrete runs (3B; swap configs for 7B)

```bash
export DATA_ROOT=./data_local
export CHECKPOINT_ROOT=./outputs

# Phase 1: data
python scripts/01_prepare_data.py --datasets gsm8k hotpotqa 2wiki finqa musique

# Phase 2: tier labels (~5h on a single 3B GPU; 5 rollouts/q over 25K q)
python scripts/02_tier_classify.py \
    --in_path  $DATA_ROOT/processed/hotpotqa_train.jsonl \
    --out_path $DATA_ROOT/tier_labels/hotpotqa.jsonl

# Phase 3: critic warm-up data (paper Appendix C)
python scripts/03_extract_warmup.py \
    --rollouts $DATA_ROOT/tier_rollouts/all.jsonl \
    --out_dir  $DATA_ROOT/critic_warmup

# Phase 3b: critic warm-up training (paper Section 3.3, Appendix D)
python scripts/04_train_critic.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --pairs $DATA_ROOT/critic_warmup/critic_warmup_pairs.jsonl \
    --out_dir outputs/critic_warmup_3b

# Phase 4: segment-level CARL PPO (paper Section 3.4, Appendix D)
python scripts/05_train_carl_ppo.py \
    --config configs/carl_3b.yaml \
    --prompts $DATA_ROOT/processed/train_pool.jsonl

# Phase 4': Search-R1 baselines (separate verl install). See
#   carl/baselines/search_r1/README.md.
python -m carl.baselines.search_r1.format_converter \
    --in $DATA_ROOT/processed/train_pool.jsonl \
    --out $DATA_ROOT/sr1/train.jsonl
python -m carl.baselines.search_r1.warmup_data \
    --carl_pairs $DATA_ROOT/critic_warmup/critic_warmup_pairs.jsonl \
    --out $DATA_ROOT/sr1/warmup.jsonl

# Phase 5: full eval (paper Table 1)
python scripts/06_eval_full.py \
    --model outputs/carl_3b/final \
    --tag carl_3b \
    --critic_ckpt outputs/critic_warmup_3b/best.pt \
    --prompts $DATA_ROOT/processed/{gsm8k,hotpotqa,2wiki,finqa,musique}_dev.jsonl
```

## Smoke tests

Two real artifacts ship under `sample_data/` so tests can run without GPUs and
reviewers can inspect the exact shapes the entry-point scripts consume:

| File | Stage | Schema |
|---|---|---|
| `sample_data/critic_warmup/critic_warmup_pairs.jsonl` | Phase 3, input to `scripts/04_train_critic.py --pairs` | `{context, V_target, segment_type, bucket, tier, prompt_mode, dataset, q_idx, rollout_idx}` |
| `sample_data/critic_warmup/warmup_stats.json` | Phase 3 sidecar (`scripts/03_extract_warmup.py`) | bucket counts |
| `sample_data/train_pool/train_pool.jsonl` | Phase 4, input to `scripts/05_train_carl_ppo.py --prompts` | `{idx, dataset, question, gold, split}` |

The critic warm-up sample is drawn from real Tier 1 / Tier 2 rollouts and
covers all four (tier, prompt_mode) buckets; forced-tool trajectories in it
contain all three segment types (invoke, assimilate, synthesize). The
training pool sample contains real prompts from each of the five datasets.

```bash
pytest tests/unit -q
pytest tests/smoke -q -m "smoke and not gpu"
pytest tests/smoke -q -m "smoke and gpu"        # requires a GPU
```

## What every paper claim maps to

| Paper element | Module |
|---|---|
| Segment decomposition (Sec. 3.1) | `carl/core/rollout.py`, `carl/core/{code,context}_block_detector.py` |
| Two-phase replacement (Sec. 3.1, App. B) | `carl/core/replacement.py` |
| Per-segment advantage Eq. 1 (Sec. 3.2, App. A) | `carl/ppo/advantages.py` |
| Critic head, MC target (Sec. 3.3) | `carl/critic/head.py`, `carl/critic/train.py` |
| Warm-up bucket construction (App. C) | `carl/critic/warmup_data.py`, `carl/data/tier_split.py` |
| PPO clipped objective Eq. 2 (Sec. 3.4) | `carl/ppo/trainer.py` |
| Hyperparameters (App. D) | `configs/carl_{3b,7b}.yaml`, `configs/sr1_*.yaml` |
| Main results (Table 1) | `scripts/06_eval_full.py` |
| Calibration (App. H) | `carl/analysis/calibration.py` |
| Error categorization (App. F) | `carl/analysis/errors.py` |
| Tool-absence resilience (App. I) | `carl/eval/resilience.py` |
| Hallucination (App. J) | `carl/eval/hallucination.py` |

Selectivity (Table 2) and hop-gains (Table 7) are computed from
`06_eval_full.py`'s per-prediction JSONL: group by `tier` / `n_hops` and
aggregate `n_tool_calls` and EM. Scale-mechanism numbers (App. G) and
capability preservation (App. I, lm-eval-harness) are reported in the
paper but not regenerated here; use lm-eval-harness for MMLU / TruthfulQA / PPL.

## Reproducing baselines (Search-R1 PPO/GRPO)

We do not reimplement Search-R1; the paper's Table 1 baseline numbers are
produced by the upstream verl-based codebase released by Jin et al. (2025).
This artifact provides:

1. A converter that rewrites the CARL training pool into Search-R1's JSONL
   format (`carl/baselines/search_r1/format_converter.py`). Both methods
   consume the same questions.
2. A converter that reuses the same `(question, reward)` warm-up pairs to
   warm Search-R1's value head (`carl/baselines/search_r1/warmup_data.py`).
   Same supervision, different boundary semantics: Search-R1's value head
   is per-token while CARL's is at segment boundaries.
3. An eval adapter that drives Search-R1's exported checkpoint through our
   eval harness so EM normalization, tool-call accounting, and per-Tier
   breakdowns are computed identically across methods
   (`carl/baselines/search_r1/eval_adapter.py`).

See `carl/baselines/search_r1/README.md` for full instructions.

The CoT+Tools prompting baselines (always-use, optional-use) are
implemented in `carl/baselines/cot_tools.py`; they require no training.

## Total compute (paper Appendix D)

| | 7B | 3B |
|---|---|---|
| CARL PPO | ~50 H100-h | ~22 H100-h |
| Search-R1 PPO | ~49 H100-h | ~21 H100-h |
| Critic warm-up | ~6 H100-h | ~4 H100-h |

Total reported numbers: ~5,500 H100-hours; full project budget including
failed runs and analysis: ~22,000 H100-hours.

## Citation

```
[anonymous for review]
```

## License

Code: MIT (see `LICENSE`). Datasets retain their original licenses
(Qwen2.5: Apache 2.0; HotpotQA: CC BY-SA 4.0; 2WikiMQA: Apache 2.0;
FinQA: MIT; Musique: CC BY 4.0; GSM8K: MIT).
