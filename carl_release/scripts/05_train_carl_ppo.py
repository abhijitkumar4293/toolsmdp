"""Phase 4 entry: configure segment-level CARL PPO training via OpenRLHF.

This is a thin entry point. The distributed runtime (Ray actor groups,
DeepSpeed strategy, vLLM engines, KL controller) lives in OpenRLHF and is
constructed by its launch script (`openrlhf.cli.train_ppo` or equivalent).
This script's job is to:

  1. Load `configs/carl_{3b,7b}.yaml` into a `PPOConfig`.
  2. Resolve the search backend and pre-build the sandbox executor.
  3. Hand both to `build_carl_ppo_trainer`, along with `generate_fn` and
     `critic_value_fn` callables that bridge OpenRLHF's runtime to CARL's
     segment loop. Those two callables must be assembled by the launch
     wrapper that owns the actor and critic Ray groups; this script alone
     cannot start training because it does not own the GPU runtime.

Use the launch wrapper at
`openrlhf/cli/train_ppo_ray.py`-equivalent and pass the values from the
YAML through `--actor`, `--critic_ckpt`, `--prompts`, `--out_dir`.
"""
import argparse
import json
from pathlib import Path

import yaml

from carl.ppo.trainer import PPOConfig, build_carl_ppo_trainer
from carl.sandbox.executor import execute_code, extract_search_query_strings
from carl.retrieval.search import MockSearch, get_bm25_search


def load_cfg(path: str) -> PPOConfig:
    raw = yaml.safe_load(Path(path).read_text())
    valid = {f for f in PPOConfig.__dataclass_fields__}
    return PPOConfig(**{k: v for k, v in raw.items() if k in valid})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="YAML, e.g. configs/carl_3b.yaml.")
    ap.add_argument("--prompts", required=True, help="Training-pool JSONL.")
    ap.add_argument("--use_mock_search", action="store_true")
    ap.add_argument("--pyserini_index", default=None,
                    help="Path to pre-built Pyserini Lucene index, or set "
                         "CARL_PYSERINI_INDEX. Required unless --use_mock_search.")
    a = ap.parse_args()

    cfg = load_cfg(a.config)

    search = MockSearch() if a.use_mock_search else get_bm25_search(index_path=a.pyserini_index)

    def execute(code: str) -> str:
        results = {q: search(q) for q in extract_search_query_strings(code)}
        return execute_code(code, search_results=results)

    prompts = [json.loads(l) for l in open(a.prompts) if l.strip()]

    # The OpenRLHF launch wrapper must construct the Ray groups, then pass
    # the resulting `generate_fn` and `critic_value_fn` to this builder.
    # See `carl/ppo/trainer.py:build_carl_ppo_trainer` for the contract.
    raise SystemExit(
        f"Loaded config={a.config} prompts={len(prompts)}. "
        "Launch via OpenRLHF: pass `cfg`, `prompts`, `execute`, `generate_fn`, "
        "`critic_value_fn`, and Ray group kwargs to `build_carl_ppo_trainer`."
    )


if __name__ == "__main__":
    main()
