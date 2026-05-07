"""Phase 2 entry: classify Tier 1/2 via 5 no-tool rollouts on the training pool."""
import argparse, json
from carl.data.tier_split import build_tier_splits
from carl.eval.hf_generator import HFGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n_rollouts", type=int, default=5)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.in_path) if l.strip()]
    gen = HFGenerator(a.model)
    def sample(prompt: str) -> str:
        return gen(prompt, stop=None, max_new_tokens=256, temperature=1.0)["text"]
    build_tier_splits(rows, sample, n_rollouts=a.n_rollouts, out_path=a.out_path)


if __name__ == "__main__":
    main()
