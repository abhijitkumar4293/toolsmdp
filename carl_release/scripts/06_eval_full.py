"""Phase 5 entry: run full evaluation harness on dev splits."""
import argparse
import json

from carl.eval.run_full import evaluate
from carl.eval.hf_generator import HFGenerator
from carl.sandbox.executor import execute_code, extract_search_query_strings
from carl.retrieval.search import MockSearch, get_bm25_search


def _load_critic(ckpt: str, model_name: str):
    """Return critic_eval(context_str) -> float, loading the warmed critic."""
    import torch
    from transformers import AutoTokenizer
    from carl.critic.head import ValueModel

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = ValueModel(model_name).cuda().eval()
    state = torch.load(ckpt, map_location="cuda")
    m.head.load_state_dict(state["head"])
    m.backbone.load_state_dict(state["backbone"])

    @torch.no_grad()
    def critic_eval(context: str) -> float:
        enc = tok(context, return_tensors="pt", truncation=True,
                  max_length=2048, add_special_tokens=False).to("cuda")
        v = m(enc["input_ids"], enc["attention_mask"])
        return float(v.item())

    return critic_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", nargs="+", required=True, help="JSONL eval splits")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", default="results/eval")
    ap.add_argument("--critic_ckpt", default=None,
                    help="Optional warmed critic; if set, V(s_0) is recorded per prediction.")
    ap.add_argument("--use_mock_search", action="store_true")
    ap.add_argument("--pyserini_index", default=None,
                    help="Path to pre-built Pyserini Lucene index, or set "
                         "CARL_PYSERINI_INDEX. Required unless --use_mock_search.")
    a = ap.parse_args()

    prompts = []
    for p in a.prompts:
        prompts.extend(json.loads(l) for l in open(p) if l.strip())

    gen = HFGenerator(a.model)
    search = MockSearch() if a.use_mock_search else get_bm25_search(index_path=a.pyserini_index)

    def execute(code: str) -> str:
        results = {q: search(q) for q in extract_search_query_strings(code)}
        return execute_code(code, search_results=results)

    critic_eval = _load_critic(a.critic_ckpt, a.model) if a.critic_ckpt else None
    summary = evaluate(prompts, gen, execute, out_dir=a.out_dir, tag=a.tag,
                       critic_eval=critic_eval)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
