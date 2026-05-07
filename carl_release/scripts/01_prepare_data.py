"""Phase 1 entry: download datasets, build training pool + dev splits."""
import argparse
from carl.data.datasets import load_split, write_jsonl, get_data_root, DATASETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    a = ap.parse_args()
    root = get_data_root()
    for ds in a.datasets:
        for sp in a.splits:
            try:
                rows = load_split(ds, sp, max_samples=a.max_samples)
                target = root / "processed" / f"{ds}_{sp}.jsonl"
                write_jsonl(rows, target)
                print(f"{ds}/{sp}: wrote {len(rows)} rows -> {target}")
            except Exception as e:
                print(f"{ds}/{sp}: SKIP ({e})")


if __name__ == "__main__":
    main()
