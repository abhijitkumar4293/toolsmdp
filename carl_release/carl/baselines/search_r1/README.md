# Search-R1 baselines (adapter, NOT a reimplementation)

We **do not** reimplement Search-R1 PPO/GRPO. We use the upstream verl-based codebase
released by Jin et al. (2025). This adapter does three things:

1. **Convert our training pool** (HotpotQA + 2WikiMQA + GSM8K) into the JSONL format
   Search-R1 expects (`<search>query</search>` tool grammar, no `<context>` blocks).
   See `format_converter.py`.

2. **Reuse our warm-up data** to warm Search-R1's value head. Both methods supervise
   their critic with binary terminal reward; only the *boundary semantics* differ.
   `warmup_data.py` rewrites our `(question, reward)` pairs into Search-R1's
   per-token-with-mask warm-up shape. Search-R1 then trains a 1-layer linear
   value head on its own backbone (separate weights from CARL).

3. **Run upstream training**, then drive the resulting checkpoint through our
   eval harness so EM, tool-call counts, and per-Tier breakdowns are computed
   identically across methods. See `eval_adapter.py`.

This way the structural differences between the two methods (boundary semantics,
gradient masking, head architecture) are preserved exactly as the original
authors implemented them.

## Setup

```
git clone https://github.com/PeterGriffinJin/Search-R1.git ~/Search-R1
cd ~/Search-R1 && git checkout <pinned commit>
pip install -e .
```

## Run paper baselines

```
python -m carl.baselines.search_r1.format_converter \
    --in $DATA_ROOT/processed/train_pool.jsonl \
    --out $DATA_ROOT/sr1/train.jsonl

python -m carl.baselines.search_r1.warmup_data \
    --carl_pairs $DATA_ROOT/critic_warmup/critic_warmup_pairs.jsonl \
    --out $DATA_ROOT/sr1/warmup.jsonl

# Then run upstream training inside the Search-R1 repo using configs/sr1_*.yaml
# as the source for hyperparameters.

# Evaluate the Search-R1 checkpoint with OUR harness (matched EM normalization,
# tool-call accounting, per-Tier breakdowns):
python scripts/06_eval_full.py \
    --model $SR1_CKPT \
    --prompts $DATA_ROOT/eval_splits/hotpotqa_dev_500.jsonl \
    --tag sr1_ppo_3b \
    --use_mock_search   # or pass --pyserini_index
```

Hyperparameters in `configs/sr1_*.yaml` mirror the paper appendix.
