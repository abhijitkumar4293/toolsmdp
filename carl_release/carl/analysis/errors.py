"""Musique error categorization (paper Appendix F).

Six categories with a written rubric, plus the aggregation that produces
the per-category counts and inter-annotator agreement.
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path


CATEGORIES = (
    "irrelevant_retrieval",
    "query_formulation",
    "assimilation_error",
    "multi_hop_reasoning",
    "commit_error",
    "other",
)

RUBRIC = """
1. irrelevant_retrieval: BM25 returned a passage that does not mention the entity / fact needed.
2. query_formulation:    Query is poorly phrased given context (e.g. mixes hops, missing entity).
3. assimilation_error:   Retrieved passage HAS the fact but <context> block dropped it.
4. multi_hop_reasoning:  All facts retrieved correctly but composed wrongly.
5. commit_error:         Context contains the answer; the final <answer> still wrong.
6. other:                Anything not above (model crashed, ill-formed answer, etc.).
"""


def categorize_errors(annotated: list[dict]) -> dict:
    """annotated rows: {q_idx, hop_count, category, annotator}."""
    overall = Counter([r["category"] for r in annotated])
    by_hop = defaultdict(Counter)
    for r in annotated:
        by_hop[r.get("hop_count", "?")].update([r["category"]])
    n = sum(overall.values())
    summary = {c: {"n": overall[c], "frac": overall[c] / max(n, 1)} for c in CATEGORIES}

    # IAA: rows annotated by both annotators
    by_q = defaultdict(dict)
    for r in annotated:
        by_q[r["q_idx"]][r["annotator"]] = r["category"]
    same, both = 0, 0
    for d in by_q.values():
        if len(d) == 2:
            both += 1
            same += int(len(set(d.values())) == 1)
    iaa = same / max(both, 1)
    return {"summary": summary, "by_hop": {h: dict(c) for h, c in by_hop.items()},
            "iaa": iaa, "n_total": n}
