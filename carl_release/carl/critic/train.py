"""Critic warm-up trainer (paper Section 3.3 / Appendix C/D).

MSE between V_phi(s) and the terminal reward, sampled uniformly across the
four warm-up buckets (Tier 1/2 x no-tool/forced-tool, paper Appendix C).
AdamW: head LR 5e-6, backbone LR 5e-7 (10x smaller). Head WD 0; backbone 0.01.
Linear warmup 100 steps, then constant. Grad clip 1.0. bf16. Verification gate
(Appendix C): AUC >= 0.70, sign accuracy >= 0.60, EV >= 0.45 on a held-out subset.
"""
from __future__ import annotations
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

from carl.critic.head import ValueModel


@dataclass
class CriticTrainConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    pairs_path: str = "data_local/critic_warmup/critic_warmup_pairs.jsonl"
    val_frac: float = 0.05
    max_len: int = 2048
    batch_size: int = 256                 # paper Appendix D
    lr_head: float = 5e-6
    lr_backbone: float = 5e-7
    weight_decay_head: float = 0.0
    weight_decay_backbone: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 2400
    eval_every: int = 200
    grad_clip: float = 1.0
    out_dir: str = "outputs/critic_warmup"
    seed: int = 1
    dtype: str = "bf16"
    attn: str = "eager"
    # verification gate (paper Appendix C)
    auc_gate: float = 0.70
    ev_gate: float = 0.45
    sign_gate: float = 0.60


class _Pairs(Dataset):
    def __init__(self, path, tokenizer, max_len: int):
        with open(path, encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["context"], truncation=True, max_length=self.max_len,
                       return_tensors="pt", add_special_tokens=False)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "v_target": torch.tensor(r["V_target"], dtype=torch.float32),
        }


def _bucket_uniform_sampler(rows: list[dict], seed: int) -> WeightedRandomSampler | None:
    """Per-row weights so each bucket has equal expected mass per batch.

    Weight of row r = 1 / count(bucket(r)). Rows missing `bucket` are weighted
    equally with one another in a synthetic "unknown" bucket; if NO row has
    `bucket`, returns None (caller falls back to plain shuffle).
    """
    buckets = [r.get("bucket", "_") for r in rows]
    if all(b == "_" for b in buckets):
        return None
    counts = Counter(buckets)
    weights = [1.0 / counts[b] for b in buckets]
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(rows),
                                  replacement=True, generator=g)


def _collate(batch, pad_id: int):
    """Left-pad so the last position is the last real token (paper Appendix D)."""
    L = max(b["input_ids"].numel() for b in batch)
    ids, mask, vt = [], [], []
    for b in batch:
        n = b["input_ids"].numel()
        pad = L - n
        ids.append(torch.cat([torch.full((pad,), pad_id, dtype=torch.long), b["input_ids"]]))
        mask.append(torch.cat([torch.zeros(pad, dtype=torch.long), b["attention_mask"]]))
        vt.append(b["v_target"])
    return {"input_ids": torch.stack(ids), "attention_mask": torch.stack(mask),
            "v_target": torch.stack(vt)}


def _explained_variance(yt, yp):
    var = (yt - yt.mean()).pow(2).mean()
    if var.item() == 0: return 0.0
    return 1.0 - (yt - yp).pow(2).mean().item() / var.item()


def _auc(scores, labels):
    """Manual ROC AUC (Mann-Whitney)."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg: return 0.5
    n = sum(int(p > ng) + 0.5 * int(p == ng) for p in pos for ng in neg)
    return n / (len(pos) * len(neg))


def train_critic(cfg: CriticTrainConfig):
    torch.manual_seed(cfg.seed)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.dtype]
    model = ValueModel(cfg.model_name, dtype=dt, attn_impl=cfg.attn).cuda()

    rows = [json.loads(l) for l in open(cfg.pairs_path, encoding="utf-8") if l.strip()]
    # Seeded shuffle BEFORE the val split so val isn't biased toward whatever
    # bucket leads the file (paper: held-out should be representative).
    import random as _rnd
    _rnd.Random(cfg.seed).shuffle(rows)
    n_val = int(len(rows) * cfg.val_frac)
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    tdir = Path(cfg.out_dir, "_splits"); tdir.mkdir(exist_ok=True)
    with open(tdir / "train.jsonl", "w") as f:
        for r in train_rows: f.write(json.dumps(r) + "\n")
    with open(tdir / "val.jsonl", "w") as f:
        for r in val_rows: f.write(json.dumps(r) + "\n")

    train_ds = _Pairs(tdir / "train.jsonl", tok, cfg.max_len)
    val_ds   = _Pairs(tdir / "val.jsonl",   tok, cfg.max_len)
    coll = lambda b: _collate(b, pad_id)

    sampler = _bucket_uniform_sampler(train_rows, cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                               sampler=sampler, shuffle=(sampler is None),
                               collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=coll)

    opt = torch.optim.AdamW([
        {"params": model.head.parameters(),     "lr": cfg.lr_head,     "weight_decay": cfg.weight_decay_head},
        {"params": model.backbone.parameters(), "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay_backbone},
    ])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, cfg.warmup_steps)))

    log_path = Path(cfg.out_dir, "train_log.jsonl"); log_path.write_text("")
    best = {"step": 0, "auc": 0.0, "ev": -1.0, "sign_acc": 0.0, "passed_gate": False}
    step = 0
    while step < cfg.max_steps:
        for batch in train_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            v = model(batch["input_ids"], batch["attention_mask"])
            loss = (v - batch["v_target"]).pow(2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); sched.step()
            step += 1
            if step % cfg.eval_every == 0 or step >= cfg.max_steps:
                m = _evaluate(model, val_loader)
                m["step"], m["train_loss"] = step, float(loss.item())
                m["passed_gate"] = (m["auc"]      >= cfg.auc_gate and
                                     m["ev"]       >= cfg.ev_gate and
                                     m["sign_acc"] >= cfg.sign_gate)
                with open(log_path, "a") as f: f.write(json.dumps(m) + "\n")
                # Best = first checkpoint to pass the FULL gate; break ties by AUC.
                better = m["passed_gate"] and (
                    not best["passed_gate"] or m["auc"] > best["auc"])
                if better:
                    best = {"step": step, **m}
                    torch.save({"head": model.head.state_dict(),
                                "backbone": model.backbone.state_dict(),
                                "cfg": cfg.__dict__, "metrics": m},
                               Path(cfg.out_dir, "best.pt"))
            if step >= cfg.max_steps: break

    Path(cfg.out_dir, "best_metrics.json").write_text(json.dumps(best, indent=2))
    return best


@torch.no_grad()
def _evaluate(model, loader):
    model.eval()
    yt, yp = [], []
    for batch in loader:
        batch = {k: v.cuda() for k, v in batch.items()}
        v = model(batch["input_ids"], batch["attention_mask"])
        yt.extend(batch["v_target"].tolist()); yp.extend(v.tolist())
    model.train()
    yt_t, yp_t = torch.tensor(yt), torch.tensor(yp)
    mse = (yt_t - yp_t).pow(2).mean().item()
    ev  = _explained_variance(yt_t, yp_t)
    auc = _auc(yp, [int(y) for y in yt])
    sign_acc = float(((yp_t >= 0.5) == (yt_t >= 0.5)).float().mean().item())
    return {"val_mse": mse, "ev": ev, "auc": auc, "sign_acc": sign_acc}

