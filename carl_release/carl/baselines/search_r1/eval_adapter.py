"""Drive a Search-R1 checkpoint through CARL's eval harness.

Search-R1's tool grammar is <search>...</search> -> <information>...</information>;
we plug in a small grammar adapter so the rollout loop and EM normalization match
exactly between methods.
"""
from __future__ import annotations
import re
from carl.core.reward import compute_reward
from carl.retrieval.search import get_bm25_search


_SR1_SYS = ("Answer the given question. You may use <search>QUERY</search>. "
            "Final answer in <answer>...</answer>.")


def run_search_r1(prompts, generate, search=None, max_calls: int = 10):
    """Search-R1 rollout: alternates <search>q</search> -> <information>r</information>.

    Generation stops at `</search>` or `</answer>`. The stop string is eaten
    by HF generate but not echoed back; we parse the OPEN tag from the text
    we just received.
    """
    search = search or get_bm25_search()
    out = []
    for ex in prompts:
        ctx = f"{_SR1_SYS}\n\nQuestion: {ex['question']}\n"
        for _ in range(max_calls):
            r = generate(prompt=ctx, stop=["</search>", "</answer>"], max_new_tokens=512)
            text = r["text"]
            ctx += text
            if "</answer>" in text:
                break
            # Extract the search query: everything between the LAST <search>
            # and end-of-context. We scan `ctx` (not just the latest chunk) so
            # this still works if <search> was opened in a previous turn and
            # the model only just emitted the closing content.
            m = re.search(r"<search>([^<]*)\Z", ctx, re.DOTALL)
            if m:
                q = m.group(1).strip()
                if q:
                    ctx += f"</search>\n<information>{search(q)}</information>\n"
                    continue
            break
        reward = compute_reward(ctx, ex["gold"], ex["dataset"])
        out.append({"context": ctx, "reward": reward, "dataset": ex["dataset"], "q_idx": ex["idx"]})
    return out
