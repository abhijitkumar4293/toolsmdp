"""Critic warmup training for ToolSMDP.

Trains a 2-layer MLP value head on top of Qwen2.5-3B-Instruct to predict
V(state) = P(reward=1) from context snapshots at segment boundaries.

Usage:
    python -m training.train_critic
    python -m training.train_critic --epochs 1 --max-samples 1000  # quick test
"""

import argparse
import json
import logging
import os
import random
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


# ── Config ─────────────────────────────────────────────────────────────────

@dataclass
class Config:
    model_name: str = MODEL_ID
    hidden_dim: int = 2048
    head_hidden: int = 1024

    # Training
    lr_head: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    epochs: int = 1
    max_seq_len: int = 2048
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # Early stopping
    early_stop_patience: int = 5  # stop after N val checks with no improvement

    # Calibration
    calibration_weight: float = 0.1
    anchor_batch_size: int = 16

    # Logging / eval
    log_every: int = 10
    val_every: int = 500
    val_max_samples: int = 2000  # subsample val set for speed

    # Paths
    data_dir: str = "data_local/critic_warmup"
    checkpoint_dir: str = "./outputs/critic_warmup"

    # Reproducibility
    seed: int = 42
    val_frac: float = 0.1

    # Optional limits (for smoke tests)
    max_samples: int | None = None

    # wandb
    wandb_project: str = "toolsmdp-critic"
    wandb_enabled: bool = False


# ── Value Head ──────────────────────────────────────────────────────────────

class ValueHead(nn.Module):
    """2-layer MLP: Linear(2048,1024) → ReLU → Linear(1024,1) → Sigmoid."""

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Upcast to float32 for loss stability (bf16 sigmoid + MSE → NaN)
        return torch.sigmoid(self.fc2(F.relu(self.fc1(x)))).float()


# ── Data ────────────────────────────────────────────────────────────────────

def load_jsonl(path: str, max_samples: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def split_by_question(warmup_rows, anchor_rows, val_frac=0.1, seed=42):
    """Split by (dataset, question_idx) — all rollouts of same question go to same split."""
    all_keys = set()
    for row in warmup_rows:
        all_keys.add((row["dataset"], row["question_idx"]))
    for row in anchor_rows:
        all_keys.add((row["dataset"], row["question_idx"]))

    all_keys = sorted(all_keys)
    rng = random.Random(seed)
    rng.shuffle(all_keys)

    n_val = int(len(all_keys) * val_frac)
    val_keys = set(all_keys[:n_val])

    def partition(rows):
        train, val = [], []
        for row in rows:
            key = (row["dataset"], row["question_idx"])
            (val if key in val_keys else train).append(row)
        return train, val

    train_w, val_w = partition(warmup_rows)
    train_a, val_a = partition(anchor_rows)
    return train_w, val_w, train_a, val_a


def collate_batch(rows, tokenizer, max_seq_len, device):
    """Tokenize, left-truncate, left-pad, return tensors."""
    contexts = [row["context"] for row in rows]
    targets = torch.tensor([row["V_target"] for row in rows], dtype=torch.float32)

    # Tokenize without truncation/padding — raw token lists
    encoded = tokenizer(contexts, add_special_tokens=False)

    # Left-truncate: keep LAST max_seq_len tokens (recent context is most predictive)
    truncated = [ids[-max_seq_len:] for ids in encoded["input_ids"]]

    # Left-pad to longest in batch
    max_len = max(len(ids) for ids in truncated)
    pad_id = tokenizer.pad_token_id

    input_ids = []
    attention_mask = []
    for ids in truncated:
        pad_len = max_len - len(ids)
        input_ids.append([pad_id] * pad_len + ids)
        attention_mask.append([0] * pad_len + [1] * len(ids))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "targets": targets.to(device),
    }


# ── Debug helpers ──────────────────────────────────────────────────────────

def debug_check_params(model, label=""):
    """Log param norms and check for NaN/Inf in model parameters."""
    nan_params, inf_params, total = 0, 0, 0
    for name, p in model.named_parameters():
        total += 1
        if torch.isnan(p).any():
            nan_params += 1
            log.error("  NaN in param %s %s", label, name)
        if torch.isinf(p).any():
            inf_params += 1
            log.error("  Inf in param %s %s", label, name)
    if nan_params or inf_params:
        log.error("DEBUG %s: %d NaN params, %d Inf params out of %d", label, nan_params, inf_params, total)
    else:
        log.info("DEBUG %s: all %d params clean", label, total)


def debug_check_grads(model, label=""):
    """Log gradient norms and check for NaN/Inf."""
    nan_grads, inf_grads, no_grad, total = 0, 0, 0, 0
    max_grad_norm = 0.0
    worst_layer = ""
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        total += 1
        if p.grad is None:
            no_grad += 1
            continue
        gn = p.grad.norm().item()
        if gn != gn:  # NaN
            nan_grads += 1
            log.error("  NaN grad: %s %s", label, name)
        elif gn == float('inf'):
            inf_grads += 1
            log.error("  Inf grad: %s %s (norm=%s)", label, name, gn)
        elif gn > max_grad_norm:
            max_grad_norm = gn
            worst_layer = name
    log.info("DEBUG %s grads: %d params, %d NaN, %d Inf, %d no_grad, max_norm=%.4f (%s)",
             label, total, nan_grads, inf_grads, no_grad, max_grad_norm, worst_layer)


def debug_check_tensor(t, name="tensor"):
    """Check a single tensor for NaN/Inf."""
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    if has_nan or has_inf:
        log.error("DEBUG %s: NaN=%s Inf=%s shape=%s norm=%s min=%s max=%s",
                  name, has_nan, has_inf, t.shape, t.norm().item(), t.min().item(), t.max().item())
    return has_nan or has_inf


# ── Forward helpers ─────────────────────────────────────────────────────────

def get_last_hidden(backbone, input_ids, attention_mask, device):
    """Forward through Qwen2Model (not CausalLM), return hidden at last real token.

    With left-padding, the last position is always the last content token.
    """
    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    return outputs.last_hidden_state[:, -1, :].to(device)  # [B, hidden_dim]


# ── Validation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(backbone, head, val_rows, val_anchors, tokenizer, config, device):
    backbone.eval()
    head.eval()

    # Subsample for speed
    if config.val_max_samples and len(val_rows) > config.val_max_samples:
        val_rows = random.sample(val_rows, config.val_max_samples)

    all_preds, all_targets = [], []
    total_loss = 0.0
    n_batches = 0

    for i in range(0, len(val_rows), config.batch_size):
        batch = collate_batch(
            val_rows[i : i + config.batch_size], tokenizer, config.max_seq_len, device
        )
        hidden = get_last_hidden(backbone, batch["input_ids"], batch["attention_mask"], device)
        v_pred = head(hidden).squeeze(-1)
        total_loss += F.mse_loss(v_pred, batch["targets"]).item()
        all_preds.extend(v_pred.cpu().tolist())
        all_targets.extend(batch["targets"].cpu().tolist())
        n_batches += 1

    # Anchor validation
    cal_preds, cal_targets = [], []
    for i in range(0, len(val_anchors), config.batch_size):
        batch = collate_batch(
            val_anchors[i : i + config.batch_size], tokenizer, config.max_seq_len, device
        )
        hidden = get_last_hidden(backbone, batch["input_ids"], batch["attention_mask"], device)
        v_pred = head(hidden).squeeze(-1)
        cal_preds.extend(v_pred.cpu().tolist())
        cal_targets.extend(batch["targets"].cpu().tolist())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # Explained variance
    target_var = np.var(targets)
    explained_var = 1.0 - np.var(targets - preds) / target_var if target_var > 0 else 0.0

    v1_mask = targets > 0.5
    v0_mask = targets < 0.5

    backbone.train()
    head.train()

    return {
        "loss": total_loss / max(n_batches, 1),
        "explained_variance": explained_var,
        "v_mean": float(preds.mean()),
        "v_std": float(preds.std()),
        "v_mean_for_v1": float(preds[v1_mask].mean()) if v1_mask.any() else 0.0,
        "v_mean_for_v0": float(preds[v0_mask].mean()) if v0_mask.any() else 0.0,
        "v_std_for_v1": float(preds[v1_mask].std()) if v1_mask.any() else 0.0,
        "v_std_for_v0": float(preds[v0_mask].std()) if v0_mask.any() else 0.0,
        "cal_loss": float(np.mean((np.array(cal_preds) - np.array(cal_targets)) ** 2))
        if cal_preds
        else 0.0,
    }


# ── Checkpoint ──────────────────────────────────────────────────────────────

def save_checkpoint(model, head, optimizer, epoch, step, metrics, config):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch{epoch}.pt")

    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "backbone_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
            "config": asdict(config),
        },
        path,
    )

    best_path = os.path.join(config.checkpoint_dir, "best.pt")
    shutil.copy2(path, best_path)
    log.info("Saved checkpoint: %s (val_loss=%.4f, expl_var=%.3f)",
             path, metrics.get("loss", -1), metrics.get("explained_variance", -1))


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa",
                        help="Attention implementation. Use 'eager' on ROCm to avoid SDPA backward NaN bug.")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--debug", action="store_true", help="Enable detailed NaN debugging logs")
    args = parser.parse_args()

    config = Config()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_samples is not None:
        config.max_samples = args.max_samples
    if args.no_wandb:
        config.wandb_enabled = False

    # Seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # ── wandb ───────────────────────────────────────────────
    if config.wandb_enabled:
        import wandb
        wandb.init(project=config.wandb_project, config=asdict(config))
    else:
        wandb = None

    # ── Model ───────────────────────────────────────────────
    log.info("Loading %s...", config.model_name)
    t0 = time.time()
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, dtype=dtype, device_map="auto", trust_remote_code=True,
        attn_implementation=args.attn,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    backbone = model.model  # Qwen2Model — skip LM head

    # Optionally freeze backbone (head-only training)
    if args.freeze_backbone:
        backbone.requires_grad_(False)
        log.info("Backbone FROZEN — training head only")
    else:
        log.info("Backbone UNFROZEN — training full model")

    # Gradient checkpointing (saves memory, trades compute)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        log.info("Gradient checkpointing ON")

    log.info("Attention implementation: %s", args.attn)

    # Debug mode: anomaly detection + detailed logging
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        log.info("DEBUG MODE: anomaly detection ON")

    # device_map="auto" spreads layers across GPU dies. Head + targets must go on
    # the same device as the final layer output. Input tensors can go anywhere —
    # HF dispatch moves them automatically.
    if hasattr(model, "hf_device_map"):
        last_dev = list(model.hf_device_map.values())[-1]
        device = torch.device(f"cuda:{last_dev}" if isinstance(last_dev, int) else last_dev)
        log.info("device_map: %s", model.hf_device_map)
    else:
        device = next(model.parameters()).device
    log.info("Model loaded in %.1fs | dtype=%s | output_device=%s", time.time() - t0, dtype, device)

    head = ValueHead(config.hidden_dim, config.head_hidden).to(device=device, dtype=dtype)
    log.info("ValueHead: %d params", sum(p.numel() for p in head.parameters()))

    # ── Data ────────────────────────────────────────────────
    log.info("Loading data from %s...", config.data_dir)
    warmup_path = os.path.join(config.data_dir, "critic_warmup_pairs.jsonl")
    anchor_path = os.path.join(config.data_dir, "critic_calibration_anchors.jsonl")

    warmup_rows = load_jsonl(warmup_path, config.max_samples)
    anchor_rows = load_jsonl(anchor_path)

    train_w, val_w, train_a, val_a = split_by_question(
        warmup_rows, anchor_rows, config.val_frac, config.seed
    )
    del warmup_rows, anchor_rows

    v1_train = sum(1 for r in train_w if r["V_target"] > 0.5)
    log.info("Train: %d warmup (%d V=1, %d V=0) + %d anchors",
             len(train_w), v1_train, len(train_w) - v1_train, len(train_a))
    log.info("Val:   %d warmup + %d anchors", len(val_w), len(val_a))

    # ── Optimizer ───────────────────────────────────────────
    if args.freeze_backbone:
        param_groups = [{"params": head.parameters(), "lr": config.lr_head}]
    else:
        param_groups = [
            {"params": head.parameters(), "lr": config.lr_head},
            {"params": backbone.parameters(), "lr": config.lr_backbone},
        ]
    optimizer = AdamW(param_groups, weight_decay=config.weight_decay)

    steps_per_epoch = len(train_w) // config.batch_size
    total_steps = steps_per_epoch * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.warmup_steps, total_steps
    )
    log.info("Steps/epoch: %d | Total steps: %d", steps_per_epoch, total_steps)

    # ── Initial validation ──────────────────────────────────
    log.info("Running initial validation (before training)...")
    val_metrics = validate(backbone, head, val_w, val_a, tokenizer, config, device)
    log.info("Initial: val_loss=%.4f expl_var=%.3f v1_mean=%.3f v0_mean=%.3f",
             val_metrics["loss"], val_metrics["explained_variance"],
             val_metrics["v_mean_for_v1"], val_metrics["v_mean_for_v0"])
    if wandb:
        wandb.log({f"val/{k}": v for k, v in val_metrics.items()} | {"step": 0, "epoch": -1})

    # ── Training ────────────────────────────────────────────
    global_step = 0
    micro_step = 0
    anchor_rng = random.Random(config.seed + 999)
    best_val_loss = float("inf")
    patience_counter = 0
    early_stopped = False
    log.info("Gradient accumulation: %d micro-batches per step (effective batch = %d)",
             args.grad_accum, config.batch_size * args.grad_accum)

    for epoch in range(config.epochs):
        epoch_rng = random.Random(config.seed + epoch)
        epoch_rng.shuffle(train_w)

        model.train()
        head.train()
        epoch_losses = []
        optimizer.zero_grad()  # zero once at start of epoch

        pbar = tqdm(
            range(0, len(train_w), config.batch_size),
            desc=f"Epoch {epoch}",
            unit="batch",
        )

        for i in pbar:
            batch_rows = train_w[i : i + config.batch_size]
            if len(batch_rows) < 2:
                continue

            batch = collate_batch(batch_rows, tokenizer, config.max_seq_len, device)

            # Debug: check inputs
            if args.debug and global_step < 3:
                debug_check_tensor(batch["input_ids"].float(), f"step{global_step}/input_ids")
                debug_check_tensor(batch["targets"], f"step{global_step}/targets")
                debug_check_params(backbone, f"step{global_step}/backbone_pre_fwd")
                debug_check_params(head, f"step{global_step}/head_pre_fwd")

            # Forward: warmup
            hidden = get_last_hidden(backbone, batch["input_ids"], batch["attention_mask"], device)

            # Debug: check hidden
            if args.debug and global_step < 3:
                debug_check_tensor(hidden, f"step{global_step}/hidden")

            v_pred = head(hidden).squeeze(-1)
            mse_loss = F.mse_loss(v_pred, batch["targets"])

            # Forward: anchors
            n_anchors = min(config.anchor_batch_size, len(train_a))
            anchor_sample = anchor_rng.sample(train_a, n_anchors)
            anchor_batch = collate_batch(anchor_sample, tokenizer, config.max_seq_len, device)
            a_hidden = get_last_hidden(
                backbone, anchor_batch["input_ids"], anchor_batch["attention_mask"], device
            )
            a_pred = head(a_hidden).squeeze(-1)
            cal_loss = F.mse_loss(a_pred, anchor_batch["targets"])

            loss = (mse_loss + config.calibration_weight * cal_loss) / args.grad_accum

            # NaN guard
            if torch.isnan(loss):
                log.error("NaN loss at step %d! mse=%.4f cal=%.4f v_pred=[%.4f,%.4f] hidden_norm=%.4f",
                          global_step, mse_loss.item(), cal_loss.item(),
                          v_pred.min().item(), v_pred.max().item(),
                          hidden.norm().item())
                if args.debug:
                    debug_check_params(backbone, "NaN/backbone")
                    debug_check_params(head, "NaN/head")
                raise RuntimeError(f"NaN loss at step {global_step}")

            # Backward (accumulate gradients)
            loss.backward()

            # Debug: check grads after backward
            if args.debug and global_step < 3:
                debug_check_grads(backbone, f"step{global_step}/backbone")
                debug_check_grads(head, f"step{global_step}/head")

            # Only step optimizer every grad_accum micro-batches
            micro_step += 1
            if micro_step % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in list(backbone.parameters()) + list(head.parameters()) if p.requires_grad],
                    config.max_grad_norm,
                )

                # Debug: log grad norm
                if args.debug and global_step < 3:
                    log.info("DEBUG step%d: grad_norm_before_clip=%.6f", global_step, grad_norm.item())

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Debug: check params after optimizer step
                if args.debug and global_step < 3:
                    debug_check_params(backbone, f"step{global_step}/backbone_post_step")
                    debug_check_params(head, f"step{global_step}/head_post_step")

            epoch_losses.append(loss.item() * args.grad_accum)  # unscaled loss
            pbar.set_postfix(loss=f"{loss.item() * args.grad_accum:.4f}")

            # Only log/validate on actual optimizer steps
            if micro_step % args.grad_accum != 0:
                global_step += 1
                continue

            # Stdout logging (always, since no wandb)
            if global_step % config.log_every == 0:
                v1_mask = batch["targets"] > 0.5
                v0_mask = batch["targets"] < 0.5
                v1_str = f"{v_pred[v1_mask].mean().item():.3f}" if v1_mask.any() else "n/a"
                v0_str = f"{v_pred[v0_mask].mean().item():.3f}" if v0_mask.any() else "n/a"
                log.info("Step %d: loss=%.4f mse=%.4f cal=%.4f v1_pred=%s v0_pred=%s",
                         global_step, loss.item(), mse_loss.item(), cal_loss.item(), v1_str, v0_str)

            # wandb logging
            if wandb and global_step % config.log_every == 0:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/mse_loss": mse_loss.item(),
                    "train/cal_loss": cal_loss.item(),
                    "train/v_mean": v_pred.mean().item(),
                    "train/v_std": v_pred.std().item(),
                    "train/lr_head": scheduler.get_last_lr()[0],
                    "train/lr_backbone": scheduler.get_last_lr()[1],
                    "epoch": epoch,
                    "step": global_step,
                }
                v1_mask = batch["targets"] > 0.5
                v0_mask = batch["targets"] < 0.5
                if v1_mask.any():
                    log_dict["train/v_pred_on_v1"] = v_pred[v1_mask].mean().item()
                if v0_mask.any():
                    log_dict["train/v_pred_on_v0"] = v_pred[v0_mask].mean().item()
                wandb.log(log_dict)

            # Mid-epoch validation
            if global_step > 0 and global_step % config.val_every == 0:
                val_metrics = validate(backbone, head, val_w, val_a, tokenizer, config, device)
                log.info(
                    "Step %d: val_loss=%.4f expl_var=%.3f v1=%.3f v0=%.3f cal=%.4f",
                    global_step, val_metrics["loss"], val_metrics["explained_variance"],
                    val_metrics["v_mean_for_v1"], val_metrics["v_mean_for_v0"],
                    val_metrics["cal_loss"],
                )
                if wandb:
                    wandb.log(
                        {f"val/{k}": v for k, v in val_metrics.items()}
                        | {"step": global_step, "epoch": epoch}
                    )

                # Early stopping + best checkpoint
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                    save_checkpoint(model, head, optimizer, epoch, global_step, val_metrics, config)
                    log.info("New best val_loss=%.4f — saved checkpoint", best_val_loss)
                else:
                    patience_counter += 1
                    log.info("No improvement (%d/%d patience)", patience_counter, config.early_stop_patience)
                    if patience_counter >= config.early_stop_patience:
                        log.info("Early stopping at step %d (best val_loss=%.4f)", global_step, best_val_loss)
                        early_stopped = True
                        break

            global_step += 1

        if early_stopped:
            break

        # ── End of epoch ────────────────────────────────────
        val_metrics = validate(backbone, head, val_w, val_a, tokenizer, config, device)

        # Save end-of-epoch checkpoint if it's the best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, head, optimizer, epoch, global_step, val_metrics, config)

        log.info(
            "Epoch %d done | train_loss=%.4f | val_loss=%.4f | expl_var=%.3f | "
            "v1_mean=%.3f | v0_mean=%.3f",
            epoch, np.mean(epoch_losses), val_metrics["loss"],
            val_metrics["explained_variance"],
            val_metrics["v_mean_for_v1"], val_metrics["v_mean_for_v0"],
        )
        if wandb:
            wandb.log(
                {f"val/{k}": v for k, v in val_metrics.items()}
                | {"step": global_step, "epoch": epoch}
            )

    log.info("Training complete. Best checkpoint: %s/best.pt (val_loss=%.4f)",
             config.checkpoint_dir, best_val_loss)
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
