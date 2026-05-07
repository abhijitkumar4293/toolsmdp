"""Reuse CARL's warm-up data to warm Search-R1's value head.

CARL's pairs are (context, V_target=R) at invoke/assimilate boundaries. Search-R1
uses a token-level value head, but its critic warm-up only needs the binary
outcome label per (question, rollout). We project CARL pairs back to the
question level (taking the question + final reward) and emit them in
Search-R1's expected shape.
"""
import argparse, json
from pathlib import Path


def convert_warmup(carl_pairs_path: str, out_path: str):
    seen = {}
    with open(carl_pairs_path) as f:
        for line in f:
            r = json.loads(line)
            key = (r["dataset"], r["q_idx"], r["rollout_idx"])
            if key not in seen:
                seen[key] = {"dataset": r["dataset"], "q_idx": r["q_idx"],
                             "rollout_idx": r["rollout_idx"], "reward": r["V_target"]}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for v in seen.values():
            f.write(json.dumps(v) + "\n")
    print(f"wrote {len(seen)} (q, rollout) reward labels -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--carl_pairs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    convert_warmup(a.carl_pairs, a.out)
