import pathlib, pickle, json, re
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

STORE = pathlib.Path(__file__).resolve().parent.parent / "faiss_store"
RRF_K = 60


def tokenize(text: str) -> list[str]:
    # lowercase words and numbers; keeps "402,836" as one token
    return re.findall(r"[a-z0-9][a-z0-9,.\-]*[a-z0-9]|[a-z0-9]", text.lower())


class Retriever:
    def __init__(self, store: pathlib.Path = STORE, mode: str = "dense"):
        assert mode in ("dense", "bm25", "hybrid")
        self.mode = mode
        self.config = json.loads((store / "config.json").read_text())
        self.index = faiss.read_index(str(store / "index.faiss"))
        with open(store / "chunks.pkl", "rb") as fh:
            self.chunks = pickle.load(fh)
        self.model = SentenceTransformer(self.config["model"])
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def _dense(self, query: str, n: int) -> list[int]:
        q = self.model.encode([query], normalize_embeddings=True)
        _, ids = self.index.search(np.asarray(q, dtype="float32"), n)
        return [int(i) for i in ids[0]]

    def _bm25(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(scores)[::-1][:n]]

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        if self.mode == "dense":
            ids = self._dense(query, k)
        elif self.mode == "bm25":
            ids = self._bm25(query, k)
        else:                                   # hybrid via RRF
            pool = 50                           # look deeper than k before fusing
            fused = {}
            for ranked in (self._dense(query, pool), self._bm25(query, pool)):
                for rank, cid in enumerate(ranked, start=1):
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            ids = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self.chunks[i] for i in ids]