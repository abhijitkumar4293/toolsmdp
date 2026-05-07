"""Smoke: warm-up data builder, search-r1 converter, sample data integrity."""
import json
from pathlib import Path
import pytest

from carl.critic.warmup_data import build_warmup_pairs
from carl.baselines.search_r1.format_converter import convert as sr1_convert
from carl.baselines.search_r1.warmup_data import convert_warmup as sr1_warmup_convert

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.smoke
def test_warmup_data_builder(tmp_path):
    fake_trs = [
        {"dataset": "gsm8k", "q_idx": 0, "rollout_idx": 0, "reward": 1.0,
         "tier": "tier2", "prompt_mode": "no_tool",
         "segments": [{"segment_type": "invoke", "context_snapshot": "Q\n```python\nprint(2)\n```"},
                      {"segment_type": "assimilate", "context_snapshot": "Q\n<context>2</context>"}]},
        {"dataset": "musique", "q_idx": 1, "rollout_idx": 0, "reward": 0.0,
         "tier": "tier1", "prompt_mode": "forced_tool",
         "segments": [{"segment_type": "invoke", "context_snapshot": "Q\n```python\nx=1\n```"}]},
    ]
    out = tmp_path / "pairs.jsonl"
    stats = build_warmup_pairs(fake_trs, out, scale="3b")
    # 2 segments from the tier2/no_tool trajectory + 1 from the tier1/forced
    assert stats["n_pairs_total"] == 3
    rows = [json.loads(l) for l in open(out)]
    assert {r["V_target"] for r in rows} == {0.0, 1.0}
    assert {r["bucket"] for r in rows} == {"tier2/no_tool", "tier1/forced_tool"}


@pytest.mark.smoke
def test_critic_warmup_sample_covers_all_buckets():
    rows = [json.loads(l) for l in open(
        ROOT / "sample_data/critic_warmup/critic_warmup_pairs.jsonl",
        encoding="utf-8")]
    buckets = {r["bucket"] for r in rows}
    assert buckets == {
        "tier2/no_tool", "tier2/forced_tool",
        "tier1/no_tool", "tier1/forced_tool",
    }
    assert {r["V_target"] for r in rows} == {0.0, 1.0}
    # Forced-tool trajectories must contribute all three segment types.
    forced_types = {r["segment_type"] for r in rows
                    if r["prompt_mode"] == "forced_tool"}
    assert forced_types == {"invoke", "assimilate", "synthesize"}


@pytest.mark.smoke
def test_train_pool_sample_schema():
    rows = [json.loads(l) for l in open(
        ROOT / "sample_data/train_pool/train_pool.jsonl", encoding="utf-8")]
    assert len(rows) > 0
    required = {"idx", "dataset", "question", "gold", "split"}
    assert required.issubset(rows[0].keys())
    assert {r["dataset"] for r in rows} == {"gsm8k", "hotpotqa", "2wiki", "finqa", "musique"}


@pytest.mark.smoke
def test_search_r1_converters(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps({"idx": 0, "dataset": "hotpotqa",
                                "question": "Q?", "gold": "A", "split": "train"}) + "\n")
    sr1_convert(str(src), str(tmp_path / "sr1.jsonl"))
    out = json.loads(open(tmp_path / "sr1.jsonl").readline())
    assert "prompt" in out and "<search>" in out["prompt"][0]["content"]
    # warmup converter
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({"context": "x", "V_target": 1.0, "segment_type": "invoke",
                                  "dataset": "gsm8k", "q_idx": 0, "rollout_idx": 0}) + "\n")
    sr1_warmup_convert(str(pairs), str(tmp_path / "wu.jsonl"))
    w = json.loads(open(tmp_path / "wu.jsonl").readline())
    assert w["reward"] == 1.0
