"""Smoke: eval harness on 3 questions with canned generator."""
import pytest
from pathlib import Path

from carl.eval.run_full import evaluate
from carl.sandbox.executor import execute_code, extract_search_query_strings
from carl.retrieval.search import MockSearch


def make_gen(answers):
    i = {"k": 0}
    def gen(prompt, stop=None, max_new_tokens=512, **kw):
        a = answers[i["k"] % len(answers)]; i["k"] += 1
        return {"text": a, "ids": [], "log_probs": []}
    return gen


@pytest.mark.smoke
def test_eval_harness(tmp_path):
    prompts = [
        {"idx": 0, "dataset": "gsm8k", "question": "what is 2+2?", "gold": "4"},
        {"idx": 1, "dataset": "hotpotqa", "question": "...", "gold": "Warsaw"},
        {"idx": 2, "dataset": "musique", "question": "...", "gold": "Edinburgh"},
    ]
    gen = make_gen(["<answer>4</answer>", "<answer>Warsaw</answer>", "<answer>Edinburgh</answer>"])
    search = MockSearch()
    def ex(code):
        rs = {q: search(q) for q in extract_search_query_strings(code)}
        return execute_code(code, search_results=rs)
    summ = evaluate(prompts, gen, ex, out_dir=str(tmp_path), tag="smoke")
    assert summ["overall"]["EM"] == 1.0
    assert (Path(tmp_path) / "smoke_metrics.json").exists()
