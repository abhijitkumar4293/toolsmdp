"""Segment-level PPO via OpenRLHF (paper Section 3.4 / Appendix D).

This module subclasses OpenRLHF's PPOTrainer. The base class handles
distributed sharding (Ray + DeepSpeed), vLLM rollout, KL bookkeeping,
optimizer construction, and checkpointing. We override only the
experience-maker so that:

  1. Token-level rollouts are replaced with segment rollouts from
     `carl.core.rollout`, which performs phase-2 replacement.
  2. The critic is queried at each segment boundary state.
  3. Each segment's per-token `advantages` tensor is the broadcast scalar
     A_k from paper Eq. 1, and `returns` is the broadcast terminal reward.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

from carl.core.rollout import rollout
from carl.core.reward import compute_reward
from carl.ppo.advantages import segment_advantages


@dataclass
class PPOConfig:
    """PPO hyperparameters (paper Appendix D)."""
    actor_model: str = "Qwen/Qwen2.5-3B-Instruct"
    critic_ckpt: str | None = None

    rollout_batch_prompts: int = 256
    n_samples_per_prompt: int = 5
    ppo_mini_batch: int = 64
    ppo_epochs: int = 4

    eps_clip: float = 0.2
    gae_lambda: float = 0.0       # one-step TD; see paper Appendix A.
    gamma: float = 1.0
    vf_coef: float = 0.5
    ent_coef: float = 1e-3
    kl_coef: float = 1e-3

    max_segments: int = 15
    invoke_max_tokens: int = 512
    assimilate_max_tokens: int = 256
    synthesize_max_tokens: int = 512

    actor_lr: float = 1e-6
    critic_value_head_lr: float = 5e-6
    critic_backbone_lr: float = 5e-7
    grad_clip: float = 1.0

    num_steps: int = 500
    eval_every: int = 25
    out_dir: str = "outputs/carl_ppo"
    seed: int = 1


def collect_segment_rollouts(prompts: list[dict], generate, execute,
                             cfg: PPOConfig) -> list[dict]:
    """Run cfg.n_samples_per_prompt segment rollouts per prompt.

    `generate(prompt, stop, max_new_tokens)` is the actor callable;
    `execute(code)` is the sandbox executor with search resolved.
    """
    out = []
    for ex in prompts:
        for j in range(cfg.n_samples_per_prompt):
            tr = rollout(
                ex["question"], generate, execute,
                max_segments=cfg.max_segments,
                invoke_max_tokens=cfg.invoke_max_tokens,
                assimilate_max_tokens=cfg.assimilate_max_tokens,
                synthesize_max_tokens=cfg.synthesize_max_tokens,
            )
            tr.reward = compute_reward(tr.full_context, ex["gold"], ex["dataset"])
            out.append({"trajectory": tr, "reward": tr.reward,
                        "dataset": ex["dataset"], "q_idx": ex["idx"], "rollout_idx": j})
    return out


def annotate_advantages(rollouts: list[dict], critic_eval: Callable) -> None:
    """Compute per-segment advantages (paper Eq. 1) on each trajectory.

    For an N-segment trajectory, V is queried at the pre-segment boundary
    states s_0..s_{N-1}. The terminal advantage uses R, not V(s_N).
    """
    for rb in rollouts:
        tr = rb["trajectory"]
        if not tr.segments:
            rb["values"], rb["advantages"] = [], []
            continue
        values = [critic_eval(s.start_context) for s in tr.segments]
        advs = segment_advantages(values, tr.reward)
        for seg, v, a in zip(tr.segments, values, advs):
            seg.value_estimate = v
            seg.value_target = tr.reward
            seg.advantage = a
        rb["values"], rb["advantages"] = values, advs


def build_carl_ppo_trainer(cfg: PPOConfig, prompts: list[dict],
                           execute_fn: Callable,
                           generate_fn: Callable,
                           critic_value_fn: Callable,
                           **openrlhf_kwargs):
    """Construct a CARLPPOTrainer (OpenRLHF subclass).

    The caller is responsible for:
      - The Ray actor groups, vLLM engines, and DeepSpeed strategy
        forwarded as **openrlhf_kwargs to the base PPOTrainer.
      - generate_fn(prompt, stop, max_new_tokens) -> {text, ids, log_probs}
      - critic_value_fn(context) -> float, V_phi at a pre-segment state.
    """
    try:
        from openrlhf.trainer import PPOTrainer
    except ImportError as e:
        raise ImportError(
            "OpenRLHF is required for paper-scale CARL training. "
            "Install with: pip install openrlhf>=0.4.0. "
            f"Original error: {e}")

    class CARLPPOTrainer(PPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._cfg = cfg
            self._prompts = prompts
            self._execute = execute_fn
            self._generate = generate_fn
            self._critic_eval = critic_value_fn

        def make_experience_batch(self, prompts_batch, **_):
            """Segment-level experience maker.

            Drives `carl.core.rollout` with the caller's generate + execute
            callables, scores binary EM reward, queries the critic at every
            pre-segment boundary, and packs segment-broadcast advantages
            into OpenRLHF Experience tensors.
            """
            rollouts = collect_segment_rollouts(prompts_batch, self._generate,
                                                self._execute, self._cfg)
            annotate_advantages(rollouts, self._critic_eval)
            return self._pack_experiences(rollouts)

        def _pack_experiences(self, rollouts):
            """One Experience per non-empty segment.

            The Experience module path differs across OpenRLHF versions; the
            constructor below matches v0.4+. Adjust if you pin a different
            release.
            """
            import torch
            from openrlhf.trainer.ppo_utils.experience_maker import Experience
            packed = []
            for rb in rollouts:
                tr = rb["trajectory"]
                R = float(rb["reward"])
                for seg in tr.segments:
                    n_gen = len(seg.generated_ids)
                    if n_gen == 0:
                        continue
                    seq = torch.tensor(seg.generated_ids, dtype=torch.long).unsqueeze(0)
                    logp = torch.tensor(seg.log_probs, dtype=torch.float32).unsqueeze(0)
                    adv = torch.full((1, n_gen), seg.advantage, dtype=torch.float32)
                    ret = torch.full((1, n_gen), R, dtype=torch.float32)
                    val = torch.full((1, n_gen), seg.value_estimate, dtype=torch.float32)
                    mask = torch.ones(1, n_gen, dtype=torch.long)
                    packed.append(Experience(
                        sequences=seq,
                        action_log_probs=logp,
                        values=val,
                        returns=ret,
                        advantages=adv,
                        attention_mask=mask,
                        action_mask=mask,
                        info={"reward": R, "segment_type": seg.segment_type},
                        kl=None,
                    ))
            return packed

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.out_dir, "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return CARLPPOTrainer(**openrlhf_kwargs)
