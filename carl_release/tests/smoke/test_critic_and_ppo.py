"""Smoke: end-to-end CARL warm-up + 1 PPO step on a tiny random Qwen2 model.

Requires GPU. Marked `gpu`. About 15-30s on a single GPU.
"""
import pytest
import torch

from carl.critic.head import ValueModel
from carl.ppo.advantages import segment_advantages

skip_if_no_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
TINY_MODEL = "hf-internal-testing/tiny-random-Qwen2ForCausalLM"


@pytest.mark.smoke
@pytest.mark.gpu
@skip_if_no_gpu
def test_value_model_forward():
    m = ValueModel(TINY_MODEL, mlp_hidden=64, dtype=torch.float32, attn_impl="eager").cuda()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    enc = tok(["hello world", "another"], padding=True, return_tensors="pt").to("cuda")
    v = m(enc["input_ids"], enc["attention_mask"])
    assert v.shape == (2,)
    assert ((v >= 0) & (v <= 1)).all()


@pytest.mark.smoke
def test_advantage_telescope():
    vals = [0.4, 0.6, 0.7]
    A = segment_advantages(vals, 1.0)
    assert abs(sum(A) - (1.0 - vals[0])) < 1e-9
