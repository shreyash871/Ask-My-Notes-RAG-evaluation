import json, pathlib, sys

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "eval" / "results"
report = json.loads((RESULTS / "generation_hybrid_k5.json").read_text(encoding="utf-8"))

wanted = set(sys.argv[1:])
for q in report["per_question"]:
    if q["id"] in wanted:
        print(f"\n=== {q['id']}  correct={q['correct']}  faithful={q['faithful']}")
        print(f"Q:      {q['question']}")
        print(f"GOLD:   {q['gold']}")
        print(f"ANSWER: {q['answer']}")
        print(f"JUDGE:  {q['reason']}")