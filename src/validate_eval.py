import json, pathlib, sys
from collections import Counter

EVAL = pathlib.Path(__file__).resolve().parent.parent / "eval" / "questions.jsonl"
REQUIRED = {"id", "company", "question", "gold_chunk_ids", "answer", "type"}

rows, ok = [], True
for n, line in enumerate(open(EVAL, encoding="utf-8"), 1):
    if not line.strip():
        continue
    try:
        row = json.loads(line)
    except json.JSONDecodeError as e:
        print(f"line {n}: bad JSON ({e})"); ok = False; continue
    missing = REQUIRED - row.keys()
    if missing:
        print(f"line {n} ({row.get('id')}): missing {missing}"); ok = False
    rows.append(row)

dupes = [i for i, c in Counter(r.get("id") for r in rows).items() if c > 1]
if dupes:
    print(f"duplicate ids: {dupes}"); ok = False

print(f"{len(rows)} rows | companies {dict(Counter(r.get('company') for r in rows))} "
      f"| types {dict(Counter(r.get('type') for r in rows))}")
print("OK" if ok else "FIX ERRORS ABOVE")
sys.exit(0 if ok else 1)