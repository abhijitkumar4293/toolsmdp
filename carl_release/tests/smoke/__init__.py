"""Smoke tests for paper artifact.

Most tests run on a tiny random Qwen2 model so they finish in seconds without
needing the real 3B/7B weights. Tests marked `gpu` are skipped if CUDA/ROCm
is unavailable.
"""
import pytest
import torch

skip_if_no_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
