import re
from typing import List
from rank_bm25 import BM25Okapi

def tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_./:-]+", text.lower())

def build_bm25_index(docs) -> BM25Okapi:
    bm25_corpus = [tokenize_for_bm25(doc.page_content) for doc in docs]
    return BM25Okapi(bm25_corpus)
