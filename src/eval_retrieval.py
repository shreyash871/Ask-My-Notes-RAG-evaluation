import json, pathlib, statistics, sys
from collections import defaultdict
from retriever import Retriever

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval" / "questions.jsonl"
RESULTS = ROOT / "eval" / "results"
RESULTS.mkdir(exist_ok=True)

K = 5


def score(retrieved_ids, gold, k):
    top = retrieved_ids[:k]
    found = [cid for cid in top if cid in gold]
    hit = 1.0 if found else 0.0
    recall = len(found) / min(k, len(gold))
    rr = 0.0
    for rank, cid in enumerate(top, start=1):
        if cid in gold:
            rr = 1.0 / rank
            break
    return {"hit": hit, "recall": recall, "rr": rr}


def mean(xs):
    return round(statistics.mean(xs), 3) if xs else 0.0


def summarize(group):
    return {"n": len(group),
            f"hit@{K}": mean([q["hit"] for q in group]),
            f"recall@{K}": mean([q["recall"] for q in group]),
            "mrr": mean([q["rr"] for q in group])}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "dense"
    print(f"MODE = {mode}")
    rows = [json.loads(l) for l in open(EVAL, encoding="utf-8") if l.strip()]
    r = Retriever(mode=mode)

    per_q = []
    for row in rows:
        ids = [h["chunk_id"] for h in r.retrieve(row["question"], k=K)]
        s = score(ids, set(row["gold_chunk_ids"]), K)
        per_q.append({**row, **s, "retrieved": ids})

    by_type, by_co = defaultdict(list), defaultdict(list)
    for q in per_q:
        by_type[q["type"]].append(q)
        by_co[q["company"]].append(q)

    report = {"mode": mode, "config": r.config, "k": K,
              "overall": summarize(per_q),
              "by_type": {t: summarize(g) for t, g in sorted(by_type.items())},
              "by_company": {c: summarize(g) for c, g in sorted(by_co.items())},
              "per_question": per_q}

    print(f"\nOVERALL  {report['overall']}")
    print("\nBY TYPE")
    for t, s in report["by_type"].items():
        print(f"  {t:8} {s}")
    print("\nBY COMPANY")
    for c, s in report["by_company"].items():
        print(f"  {c:8} {s}")
    print("\nMISSES")
    for q in per_q:
        if q["hit"] == 0:
            print(f"  {q['id']}  {q['question'][:70]}")

    out = RESULTS / f"retrieval_{mode}_cs{r.config['chunk_size']}_k{K}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out.name}")


if __name__ == "__main__":
    main()