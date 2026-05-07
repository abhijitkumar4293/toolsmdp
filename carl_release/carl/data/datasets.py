"""Dataset download and unified JSONL format.

Outputs records with: {idx, dataset, question, gold, split, [meta...]}
DATA_ROOT env var controls cache location.
"""
import json
import os
import re
import urllib.request
from pathlib import Path

DATASETS = ("gsm8k", "hotpotqa", "2wiki", "finqa", "musique")


def get_data_root() -> Path:
    p = Path(os.environ.get("DATA_ROOT", "./data_local"))
    p.mkdir(parents=True, exist_ok=True)
    (p / "processed").mkdir(exist_ok=True)
    (p / "eval_splits").mkdir(exist_ok=True)
    return p


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _gsm8k(split, max_samples):
    from datasets import load_dataset
    hf_split = "test" if split in ("dev", "test") else "train"
    ds = load_dataset("openai/gsm8k", "main", split=hf_split)
    if max_samples: ds = ds.select(range(min(max_samples, len(ds))))
    out = []
    for i, ex in enumerate(ds):
        m = re.search(r"####\s*(.+)", ex["answer"])
        gold = m.group(1).strip() if m else ex["answer"].strip()
        out.append({"idx": i, "dataset": "gsm8k", "split": split,
                    "question": ex["question"], "gold": gold})
    return out


def _hotpotqa(split, max_samples):
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split="train" if split == "train" else "validation")
    if max_samples: ds = ds.select(range(min(max_samples, len(ds))))
    out = []
    for i, ex in enumerate(ds):
        out.append({"idx": i, "dataset": "hotpotqa", "split": split,
                    "question": ex["question"], "gold": ex["answer"], "type": ex.get("type")})
    return out


def _2wiki(split, max_samples):
    from datasets import load_dataset
    ds = load_dataset("scholarly-shadows-syndicate/2WikiMultiHopQA",
                      split="train" if split == "train" else "validation",
                      trust_remote_code=True)
    if max_samples: ds = ds.select(range(min(max_samples, len(ds))))
    out = []
    for i, ex in enumerate(ds):
        out.append({"idx": i, "dataset": "2wiki", "split": split,
                    "question": ex["question"], "gold": ex["answer"]})
    return out


def _finqa(split, max_samples):
    # Upstream FinQA release on GitHub. The repo provides train/dev/test;
    # we use train and test. There is no separate dev split in the release.
    urls = {
        "train": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
        "dev":   "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
        "test":  "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
    }
    raw = json.loads(urllib.request.urlopen(urls[split], timeout=60).read())
    if max_samples: raw = raw[:max_samples]
    out = []
    for i, ex in enumerate(raw):
        parts = []
        if ex.get("pre_text"): parts.append("\n".join(ex["pre_text"]))
        if ex.get("table"):
            rows = []
            for k, row in enumerate(ex["table"]):
                rows.append("| " + " | ".join(str(c) for c in row) + " |")
                if k == 0:
                    rows.append("|" + " --- |" * len(row))
            parts.append("\n".join(rows))
        if ex.get("post_text"): parts.append("\n".join(ex["post_text"]))
        parts.append(ex["qa"]["question"])
        q = "\n\n".join(parts)
        out.append({"idx": i, "dataset": "finqa", "split": split,
                    "question": q, "gold": str(ex["qa"]["exe_ans"])})
    return out


def _musique(split, max_samples):
    from datasets import load_dataset
    ds = load_dataset("bdsaglam/musique",
                      split="train" if split == "train" else "validation")
    if max_samples: ds = ds.select(range(min(max_samples, len(ds))))
    out = []
    for i, ex in enumerate(ds):
        decomp = ex.get("question_decomposition")
        out.append({"idx": i, "dataset": "musique", "split": split,
                    "question": ex["question"], "gold": ex["answer"],
                    "n_hops": len(decomp) if decomp else None})
    return out


_LOADERS = {
    "gsm8k": _gsm8k, "hotpotqa": _hotpotqa, "2wiki": _2wiki,
    "finqa": _finqa, "musique": _musique,
}


def load_split(dataset: str, split: str = "dev", max_samples: int | None = None):
    """Load and unify a dataset split. `split` accepts 'train', 'dev', 'test'."""
    return _LOADERS[dataset](split, max_samples)
