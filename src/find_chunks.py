import sys, pickle, pathlib

STORE = pathlib.Path(__file__).resolve().parent.parent / "faiss_store"
chunks = pickle.load(open(STORE / "chunks.pkl", "rb"))

# usage: python src/find_chunks.py AMZN "AWS" "segment"
company = sys.argv[1].upper()
terms = [t.lower() for t in sys.argv[2:]]

for c in chunks:
    if c["company"] != company:
        continue
    text = c["text"].lower()
    if all(t in text for t in terms):
        pos = text.find(terms[0])
        start = max(0, pos - 150)
        snippet = c["text"][start:pos + 250].replace("\n", " ")
        print(f"#{c['chunk_id']:<5} Item {c['section']:<3} ...{snippet}...\n")