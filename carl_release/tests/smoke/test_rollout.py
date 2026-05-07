"""Smoke: full segment rollout against a tiny in-memory generator + mock search."""
import pytest
from carl.core.rollout import rollout
from carl.core.reward import compute_reward
from carl.sandbox.executor import execute_code, extract_search_query_strings
from carl.retrieval.search import MockSearch


def make_canned_generator(scripts):
    """scripts: list of dicts {prompt_substr -> response} drawn in order."""
    state = {"i": 0}
    def gen(prompt: str, stop=None, max_new_tokens=512, **kw):
        s = scripts[state["i"]]
        state["i"] = min(state["i"] + 1, len(scripts) - 1)
        return {"text": s, "ids": list(range(len(s))), "log_probs": [-1.0] * len(s)}
    return gen


@pytest.mark.smoke
def test_one_invoke_then_commit():
    gen = make_canned_generator([
        "Let me search.\n```python\nprint(search('Marie Curie birthplace'))\n```",
        "<context>Marie Curie born Warsaw</context>",
        "<answer>Warsaw</answer>",
    ])
    search = MockSearch({"Marie Curie birthplace": "Marie Curie was born in Warsaw, Poland."})

    def execute(code: str) -> str:
        results = {q: search(q) for q in extract_search_query_strings(code)}
        return execute_code(code, search_results=results)

    tr = rollout("Where was Marie Curie born?", gen, execute)
    assert tr.total_tool_calls >= 1
    assert "Warsaw" in tr.full_context
    assert compute_reward(tr.full_context, "Warsaw", "musique") == 1.0


@pytest.mark.smoke
def test_no_invoke_path():
    gen = make_canned_generator(["<answer>4</answer>"])
    tr = rollout("What is 2+2?", gen, lambda c: "")
    assert tr.total_tool_calls == 0
    assert "4" in tr.full_context
