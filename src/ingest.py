import json, pathlib, pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from parse import html_to_text, extract_sections, DATA

ROOT = pathlib.Path(__file__).resolve().parent.parent
STORE = ROOT / "faiss_store"
STORE.mkdir(exist_ok=True)

CHUNK_SIZE = 1500
OVERLAP = 200
MODEL_NAME = "all-MiniLM-L6-v2"

SECTION_NAMES = {"1": "Business", "1A": "Risk Factors",
                 "7": "MD&A", "8": "Financial Statements"}

COMPANY_NAMES = {"AAPL": "Apple", "MSFT": "Microsoft",
                 "GOOGL": "Alphabet Google", "AMZN": "Amazon",
                 "META": "Meta Facebook"}


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split on paragraph breaks, packing paragraphs up to `size` chars."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= size:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            if cur:
                chunks.append(cur)
            # carry the tail of the previous chunk forward as overlap
            tail = cur[-overlap:] if cur else ""
            cur = f"{tail}\n\n{p}" if tail else p
            while len(cur) > size:          # a single huge paragraph
                chunks.append(cur[:size])
                cur = cur[size - overlap:]
    if cur:
        chunks.append(cur)
    return chunks


def build():
    records = []
    for f in sorted(DATA.glob("*.htm")):
        ticker, _, year = f.stem.split("_")        # AAPL_10K_2025
        sections = extract_sections(html_to_text(f))
        for item, body in sections.items():
            for piece in chunk_text(body, CHUNK_SIZE, OVERLAP):
                if len(piece) < 200:               # drop page-number debris
                    continue
                header = f"[{COMPANY_NAMES[ticker]} ({ticker}) {year} | Item {item} {SECTION_NAMES[item]}]"
                records.append({
                    "chunk_id": len(records),
                    "company": ticker,
                    "year": int(year),
                    "section": item,
                    "text": f"{header}\n{piece}",
                })

    print(f"{len(records)} chunks. Embedding...")
    model = SentenceTransformer(MODEL_NAME)
    vecs = model.encode([r["text"] for r in records], batch_size=64,
                        show_progress_bar=True, normalize_embeddings=True)

    index = faiss.IndexFlatIP(vecs.shape[1])       # inner product = cosine (normalized)
    index.add(np.asarray(vecs, dtype="float32"))

    faiss.write_index(index, str(STORE / "index.faiss"))
    with open(STORE / "chunks.pkl", "wb") as fh:
        pickle.dump(records, fh)
    with open(STORE / "config.json", "w") as fh:
        json.dump({"chunk_size": CHUNK_SIZE, "overlap": OVERLAP,
                   "model": MODEL_NAME, "n_chunks": len(records)}, fh, indent=2)
    print("Saved to faiss_store/")


if __name__ == "__main__":
    build()