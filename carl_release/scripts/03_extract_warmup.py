"""Phase 3 entry: build the 4-bucket critic warm-up dataset (paper Appendix C)."""
import argparse, json
from pathlib import Path
from carl.critic.warmup_data import build_warmup_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True,
                    help="JSONL of rollouts with reward + tier + prompt_mode + segments[].context_snapshot")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scale", choices=["3b", "7b"], default="3b")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    trs = [json.loads(l) for l in open(a.rollouts) if l.strip()]
    stats = build_warmup_pairs(trs, out / "critic_warmup_pairs.jsonl",
                                scale=a.scale, seed=a.seed)
    (out / "warmup_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"scale={a.scale}  pairs={stats['n_pairs_total']}")
    for k in stats["kept_trajectories"]:
        print(f"  {k:20s}  trajectories={stats['kept_trajectories'][k]:6d}"
              f"  pairs={stats['kept_pairs'][k]:7d}")


if __name__ == "__main__":
    main()
