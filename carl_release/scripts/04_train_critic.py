"""Phase 3 entry: train critic head with MSE warm-up."""
import argparse
from carl.critic.train import train_critic, CriticTrainConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out_dir", default="outputs/critic_warmup")
    ap.add_argument("--max_steps", type=int, default=2400)
    ap.add_argument("--batch_size", type=int, default=256)
    a = ap.parse_args()
    cfg = CriticTrainConfig(model_name=a.model, pairs_path=a.pairs,
                             out_dir=a.out_dir, max_steps=a.max_steps,
                             batch_size=a.batch_size)
    best = train_critic(cfg)
    print(best)


if __name__ == "__main__":
    main()
