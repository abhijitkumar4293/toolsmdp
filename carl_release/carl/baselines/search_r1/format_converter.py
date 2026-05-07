"""Convert CARL-format training rows into Search-R1 (verl) JSONL shape."""
import argparse, json
from pathlib import Path
from carl.data.datasets import read_jsonl


SR1_SYSTEM = (
    "Answer the given question. You may use <search>QUERY</search> to retrieve "
    "Wikipedia passages, which will be returned inside <information>...</information>. "
    "After retrieval, give your final answer inside <answer>...</answer>."
)


def convert(in_path: str, out_path: str):
    rows = read_jsonl(in_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps({
                "data_source": r["dataset"],
                "prompt": [
                    {"role": "system", "content": SR1_SYSTEM},
                    {"role": "user", "content": r["question"]},
                ],
                "ability": "qa",
                "reward_model": {"style": "rule", "ground_truth": r["gold"]},
                "extra_info": {"split": r.get("split", "train"), "index": r["idx"]},
            }) + "\n")
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    a = ap.parse_args()
    convert(a.in_path, a.out_path)
