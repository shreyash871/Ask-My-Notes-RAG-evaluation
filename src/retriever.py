import pathlib, pickle, json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

STORE = pathlib.Path(__file__).resolve().parent.parent / "faiss_store"


class Retriever:
    def __init__(self, store: pathlib.Path = STORE):
        self.config = json.loads((store / "config.json").read_text())
        self.index = faiss.read_index(str(store / "index.faiss"))
        with open(store / "chunks.pkl", "rb") as fh:
            self.chunks = pickle.load(fh)
        self.model = SentenceTransformer(self.config["model"])

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        q = self.model.encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), k)
        return [{**self.chunks[i], "score": float(s)}
                for s, i in zip(scores[0], ids[0])]


if __name__ == "__main__":
    r = Retriever()
    queries = [
        "What was Apple's total net sales?",
        "What are Meta's risks related to regulation of user data?",
        "How does Amazon describe AWS?",
        "What does Microsoft say about competition in cloud?",
        "What were Alphabet's advertising revenues?",
    ]
    for q in queries:
        print(f"\n=== {q}")
        for hit in r.retrieve(q, k=3):
            preview = hit["text"][:110].replace("\n", " ")
            print(f"  {hit['score']:.3f}  #{hit['chunk_id']:<5} {preview}")