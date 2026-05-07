"""Minimal HuggingFace-backed generate(prompt, stop, max_new_tokens) callable.

Matches the GenFn signature consumed by carl.core.rollout. For training,
plug a vLLM client into the same signature. This file is the eval backend.
"""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFGenerator:
    def __init__(self, model_name: str, device: str = "cuda", dtype=torch.bfloat16):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, prompt: str, stop: list[str] | None = None,
                 max_new_tokens: int = 512, temperature: float = 0.0):
        inputs = self.tok(prompt, return_tensors="pt", truncation=True,
                          max_length=4096).to(self.device)
        n_in = inputs["input_ids"].size(1)
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=temperature or 1.0,
            pad_token_id=self.tok.pad_token_id, return_dict_in_generate=True,
            output_scores=True,
        )
        ids = out.sequences[0, n_in:].tolist()
        text = self.tok.decode(ids, skip_special_tokens=True)
        n_keep = len(ids)
        if stop:
            cut = min((text.find(s) + len(s) for s in stop if s in text), default=-1)
            if cut >= 0:
                text = text[:cut]
                # Find the smallest prefix of `ids` whose decoding contains
                # the trimmed text. This keeps ids aligned with `out.scores`.
                for k in range(1, len(ids) + 1):
                    if self.tok.decode(ids[:k], skip_special_tokens=True).startswith(text):
                        n_keep = k
                        break
                ids = ids[:n_keep]
        if out.scores:
            scores = out.scores[:n_keep]
            lps = [float(torch.log_softmax(s[0], dim=-1)[t].item())
                   for t, s in zip(ids, scores)]
        else:
            lps = []
        return {"text": text, "ids": ids, "log_probs": lps}
