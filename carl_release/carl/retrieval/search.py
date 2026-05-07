"""BM25 retrieval over Wikipedia (Pyserini), plus a MockSearch for tests.

The Pyserini index must already be built and present on disk; this module
does not download it. Set CARL_PYSERINI_INDEX to the local Lucene index
directory before training or eval, or pass `index_path=` explicitly.
"""
import json
import os
from pathlib import Path

_searcher_cache: dict = {}


def get_bm25_search(top_k: int = 3, index_path: str | None = None):
    """Return query(str) -> str backed by a pre-built Pyserini Lucene index.

    Raises FileNotFoundError if the index isn't on disk. Cached per
    index_path so multiple paths in the same process don't share a
    single searcher.
    """
    path = index_path or os.environ.get("CARL_PYSERINI_INDEX")
    if not path:
        raise RuntimeError(
            "Pyserini index path not provided. Set CARL_PYSERINI_INDEX or pass "
            "index_path=. The index must be pre-built (see project README).")
    if path in _searcher_cache:
        return _searcher_cache[path]
    if not Path(path).is_dir():
        raise FileNotFoundError(
            f"Pyserini index not found at {path}. Build or download it before "
            "running training/eval; auto-download is not performed.")

    from pyserini.search.lucene import LuceneSearcher
    s = LuceneSearcher(path)

    def q(query: str, k: int = top_k) -> str:
        hits = s.search(query, k=k)
        if not hits:
            return "No relevant results found."
        out = []
        for r, h in enumerate(hits, 1):
            d = json.loads(s.doc(h.docid).raw())
            out.append(f"[{r}] {d.get('title','')}\n{d.get('contents','')}")
        return "\n\n".join(out)

    _searcher_cache[path] = q
    return q


class MockSearch:
    """Pre-loaded search responses keyed by query string. For tests."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "MockSearch":
        d = {}
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    d[r["query"]] = r["result"]
        return cls(d)

    def __call__(self, query: str, k: int = 3) -> str:
        return self.responses.get(query, "No relevant results found.")
