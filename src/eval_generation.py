import json, os, pathlib, sys, time, re, statistics
from collections import defaultdict
from dotenv import load_dotenv
from groq import Groq
from retriever import Retriever
from groq import RateLimitError, APIConnectionError

load_dotenv()
ROOT = pathlib.Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval" / "questions.jsonl"
RESULTS = ROOT / "eval" / "results"

GEN_MODEL = "openai/gpt-oss-120b"
JUDGE_MODEL = "qwen/qwen3.8-27b"
K = 5

client = Groq(api_key=os.environ["GROQ_API_KEY"])

ANSWER_PROMPT = """Answer the question using ONLY the context below from SEC 10-K filings.
If the context does not contain the answer, reply exactly: I don't know.
Give figures with their units. Be concise.

Context:
{context}

Question: {question}"""

JUDGE_PROMPT = """You are grading a RAG system. Reply with JSON only.

Question: {question}
Reference answer: {gold}

Retrieved context:
{context}

System answer: {answer}

Return:
{{"correct": true/false,   // does the system answer match the reference answer in substance (numbers must match)?
  "faithful": true/false,  // is EVERY claim in the system answer supported by the retrieved context?
  "abstained": true/false, // did the system say it doesn't know?
  "reason": "<one sentence>"}}"""


def chat(model, prompt, retries=5):
    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            return resp, time.perf_counter() - t0
        except (RateLimitError, APIConnectionError) as e:
            wait = 2 ** attempt * 5
            print(f"    retry in {wait}s ({type(e).__name__})")
            time.sleep(wait)
    raise RuntimeError("Groq call failed after retries")


def parse_judge(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(re.sub(r"//.*", "", m.group(0)))
    except Exception:
        return {"correct": False, "faithful": False, "abstained": False,
                "reason": "judge output unparseable"}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
    rows = [json.loads(l) for l in open(EVAL, encoding="utf-8") if l.strip()]
    r = Retriever(mode=mode)
    per_q = []

    for row in rows:
        print(f"{row['id']} ...")
        t0 = time.perf_counter()
        chunks = r.retrieve(row["question"], k=K)
        t_ret = time.perf_counter() - t0
        context = "\n\n---\n\n".join(c["text"] for c in chunks)

        resp, t_gen = chat(GEN_MODEL, ANSWER_PROMPT.format(
            context=context, question=row["question"]))
        answer = resp.choices[0].message.content.strip()

        jresp, _ = chat(JUDGE_MODEL, JUDGE_PROMPT.format(
            question=row["question"], gold=row["answer"],
            context=context, answer=answer))
        verdict = parse_judge(jresp.choices[0].message.content)

        per_q.append({
            "id": row["id"], "company": row["company"], "type": row["type"],
            "question": row["question"], "gold": row["answer"], "answer": answer,
            **verdict,
            "latency_retrieval_s": round(t_ret, 3),
            "latency_generation_s": round(t_gen, 3),
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        })
        time.sleep(2)                             # stay under free-tier limits

    def rate(group, key):
        return round(sum(1 for q in group if q.get(key)) / len(group), 3)

    def summarize(g):
        answered = [q for q in g if not q.get("abstained")]
        return {"n": len(g),
                "correct": rate(g, "correct"),
                "faithful": rate(g, "faithful"),
                "abstain": rate(g, "abstained"),
                "hallucination": round(sum(1 for q in answered if not q.get("faithful"))
                                       / max(1, len(answered)), 3)}

    by_type = defaultdict(list)
    for q in per_q:
        by_type[q["type"]].append(q)

    lat = [q["latency_retrieval_s"] + q["latency_generation_s"] for q in per_q]
    report = {
        "mode": mode, "gen_model": GEN_MODEL, "judge_model": JUDGE_MODEL, "k": K,
        "overall": summarize(per_q),
        "by_type": {t: summarize(g) for t, g in sorted(by_type.items())},
        "latency_s": {"p50": round(statistics.median(lat), 3),
                      "p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 3)},
        "avg_prompt_tokens": round(statistics.mean(q["prompt_tokens"] for q in per_q)),
        "avg_completion_tokens": round(statistics.mean(q["completion_tokens"] for q in per_q)),
        "per_question": per_q,
    }

    print(f"\nOVERALL  {report['overall']}")
    for t, s in report["by_type"].items():
        print(f"  {t:8} {s}")
    print(f"LATENCY  {report['latency_s']}")
    print(f"TOKENS   prompt {report['avg_prompt_tokens']}  completion {report['avg_completion_tokens']}")
    print("\nPROBLEMS")
    for q in per_q:
        if not q.get("correct") or not q.get("faithful"):
            print(f"  {q['id']}  correct={q.get('correct')} faithful={q.get('faithful')}  {q.get('reason','')[:80]}")

    out = RESULTS / f"generation_{mode}_k{K}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out.name}")


if __name__ == "__main__":
    main()