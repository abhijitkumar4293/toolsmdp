"""Extract critic warmup training data from tier v5 rollouts.

Combines Steps 3.1 (warmup pairs), 3.2 (calibration anchors), and 3.3 (contrastive pairs).

Uses context_snapshot stored at each segment boundary (saved during tier v5 generation).
Only extracts invoke and assimilate boundaries — initial and synthesize are dropped.

Usage:
    python -m analysis.extract_critic_pairs \
        --input-dir downloads/tier_v5/artifacts/outputs \
        --output-dir data_local/critic_warmup
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATASETS = ["gsm8k", "hotpotqa", "2wiki", "finqa", "musique"]


def extract_pairs_from_file(path: Path, dataset: str) -> dict:
    """Process one dataset file. Returns warmup pairs, anchors, and contrastive info."""

    warmup_pairs = []
    easy_anchors = []
    hard_anchors = []
    contrastive_questions = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            question = row["question"]
            question_idx = row["question_idx"]
            tier = row["tier"]
            tool_rollouts = row["tool_rollouts"]
            no_tool_rollouts = row.get("no_tool_rollouts", [])

            # --- Per-rollout pairs (Step 3.1) ---
            for list_idx, rollout in enumerate(tool_rollouts):
                reward = rollout["reward"]
                if reward is None:
                    continue

                for seg in rollout["segments"]:
                    seg_type = seg["type"]

                    # Only invoke and assimilate boundaries
                    if seg_type not in ("invoke", "assimilate"):
                        continue

                    snapshot = seg.get("context_snapshot")
                    if not snapshot:
                        continue

                    warmup_pairs.append({
                        "context": snapshot,
                        "V_target": reward,
                        "segment_type": seg_type,
                        "dataset": dataset,
                        "question_idx": question_idx,
                        "rollout_idx": list_idx,
                    })

            # --- Calibration anchors (Step 3.2) ---
            tool_rewards = [r["reward"] for r in tool_rollouts if r["reward"] is not None]
            no_tool_rewards = [r["reward"] for r in no_tool_rollouts if r["reward"] is not None]

            # Easy anchor: Tier 2 + all no-tool rollouts correct
            if tier == "tier2" and no_tool_rewards and all(r > 0 for r in no_tool_rewards):
                easy_anchors.append({
                    "context": question,
                    "V_target": 1.0,
                    "anchor_type": "easy",
                    "dataset": dataset,
                    "question_idx": question_idx,
                })

            # Hard anchor: Tier 1 + ALL tool rollouts failed
            if tier == "tier1" and tool_rewards and all(r == 0 for r in tool_rewards):
                hard_anchors.append({
                    "context": question,
                    "V_target": 0.0,
                    "anchor_type": "hard",
                    "dataset": dataset,
                    "question_idx": question_idx,
                })

            # --- Contrastive pairs (Step 3.3) ---
            r1_rollouts = [r for r in tool_rollouts if r["reward"] is not None and r["reward"] > 0]
            r0_rollouts = [r for r in tool_rollouts if r["reward"] is not None and r["reward"] == 0]

            if r1_rollouts and r0_rollouts:
                contrastive_questions.append({
                    "dataset": dataset,
                    "question_idx": question_idx,
                    "question": question,
                    "num_r1": len(r1_rollouts),
                    "num_r0": len(r0_rollouts),
                })

    return {
        "warmup_pairs": warmup_pairs,
        "easy_anchors": easy_anchors,
        "hard_anchors": hard_anchors,
        "contrastive_questions": contrastive_questions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
                        default=Path("downloads/tier_v5/artifacts/outputs"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data_local/critic_warmup"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_warmup = []
    all_easy = []
    all_hard = []
    all_contrastive = []

    for ds in DATASETS:
        path = args.input_dir / f"{ds}_tier_splits.jsonl"
        if not path.exists():
            log.warning(f"Skipping {ds}: {path} not found")
            continue

        log.info(f"Processing {ds}...")
        result = extract_pairs_from_file(path, ds)

        n_warmup = len(result["warmup_pairs"])
        n_easy = len(result["easy_anchors"])
        n_hard = len(result["hard_anchors"])
        n_contrastive = len(result["contrastive_questions"])

        log.info(f"  {ds}: {n_warmup:,} warmup, "
                 f"{n_easy:,} easy anchors, {n_hard:,} hard anchors, "
                 f"{n_contrastive:,} contrastive Qs")

        all_warmup.extend(result["warmup_pairs"])
        all_easy.extend(result["easy_anchors"])
        all_hard.extend(result["hard_anchors"])
        all_contrastive.extend(result["contrastive_questions"])

    # Write outputs
    warmup_path = args.output_dir / "critic_warmup_pairs.jsonl"
    with open(warmup_path, "w", encoding="utf-8") as f:
        for pair in all_warmup:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(all_warmup):,} warmup pairs to {warmup_path}")

    anchors_path = args.output_dir / "critic_calibration_anchors.jsonl"
    with open(anchors_path, "w", encoding="utf-8") as f:
        for a in all_easy + all_hard:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(all_easy):,} easy + {len(all_hard):,} hard anchors to {anchors_path}")

    contrastive_path = args.output_dir / "critic_contrastive_questions.jsonl"
    with open(contrastive_path, "w", encoding="utf-8") as f:
        for c in all_contrastive:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(all_contrastive):,} contrastive questions to {contrastive_path}")

    # Summary
    v1_count = sum(1 for p in all_warmup if p["V_target"] > 0)
    v0_count = sum(1 for p in all_warmup if p["V_target"] == 0)
    log.info(f"\nSummary:")
    log.info(f"  Total warmup pairs: {len(all_warmup):,}")
    log.info(f"    V=1 pairs: {v1_count:,} ({100*v1_count/len(all_warmup):.1f}%)")
    log.info(f"    V=0 pairs: {v0_count:,} ({100*v0_count/len(all_warmup):.1f}%)")
    log.info(f"  Calibration anchors: {len(all_easy):,} easy + {len(all_hard):,} hard")
    log.info(f"  Contrastive questions: {len(all_contrastive):,}")

    # Per-segment-type breakdown
    type_counts = {}
    for p in all_warmup:
        t = p["segment_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    log.info(f"  Segment type distribution:")
    for t, c in sorted(type_counts.items()):
        log.info(f"    {t}: {c:,} ({100*c/len(all_warmup):.1f}%)")

    # Per-dataset breakdown
    log.info(f"\n  Per-dataset breakdown:")
    ds_counts = {}
    for p in all_warmup:
        d = p["dataset"]
        v = p["V_target"]
        key = (d, "V=1" if v > 0 else "V=0")
        ds_counts[key] = ds_counts.get(key, 0) + 1
    for ds in DATASETS:
        v1 = ds_counts.get((ds, "V=1"), 0)
        v0 = ds_counts.get((ds, "V=0"), 0)
        total = v1 + v0
        log.info(f"    {ds}: {total:,} ({v1:,} V=1, {v0:,} V=0, "
                 f"ratio={v1/total:.2f})" if total else f"    {ds}: 0")


if __name__ == "__main__":
    main()
