# Ask-My-Notes: RAG + Evaluation Harness

A retrieval-augmented QA system, benchmarked on SEC 10-K annual filings, with an evaluation harness that measures **retrieval quality**, **hallucination rate**, and **latency/token cost**.

The point of this project is not that the RAG pipeline runs. It's that every claim about how well it works is backed by a number, and that the numbers themselves were audited.

**🔗 Live demo:** [ask-my-notes-rag-evaluation-ask-10k.streamlit.app](https://ask-my-notes-rag-evaluation-ask-10k.streamlit.app)

**Try asking:**
- *What were AWS net sales in fiscal year 2025?*
- *Which specific AI regulation does Microsoft cite as a risk to its European business?*
- *What was Nvidia's revenue in 2025?* (not in the corpus, so it should say "I don't know")

Every answer shows the retrieved source chunks it was built from. The app runs on a free tier and sleeps when idle, so the first question after a pause can take up to a minute.

---

## Results at a glance

| | Value |
|---|---|
| Retrieval hit@5 (hybrid) | **0.933** |
| Answer correctness | **93%** (28/30) |
| Hallucination rate (answered questions) | **0%** |
| Latency p50 / p95 | 0.85s / 1.25s |
| Tokens per query | ~1,660 prompt, ~168 completion |

---

## Corpus

Five most recent 10-K filings, fetched reproducibly from the SEC EDGAR API by `src/fetch_filings.py`:

| Company | Ticker | Fiscal year end |
|---|---|---|
| Apple | AAPL | last Saturday of September |
| Microsoft | MSFT | June 30 |
| Alphabet | GOOGL | December 31 |
| Amazon | AMZN | December 31 |
| Meta | META | December 31 |

Five companies in the **same sector** was a deliberate choice: their filings use near-identical language ("risk factors", "revenue recognition"), so the retriever has to tell *Apple's* supply-chain risk apart from *Microsoft's*. That's a harder and more realistic test than a corpus where every document covers a different topic.

**Scope:** only Items 1 (Business), 1A (Risk Factors), 7 (MD&A) and 8 (Financial Statements) are indexed. The remaining ~70% of a 10-K (exhibit lists, signatures, XBRL schema, auditor consents) adds noise that competes for top-k slots without answering questions anyone asks. The trade-off is that questions about excluded sections are unanswerable by design.

Filings are not committed to the repo; `fetch_filings.py` reproduces the exact corpus in about 20 seconds.

---

## Pipeline

```
EDGAR API ──► HTML ──► clean text ──► Items 1/1A/7/8 ──► 1,500-char chunks ──► embeddings + BM25
                                                                                       │
                              question ──► dense + BM25 ──► reciprocal rank fusion ──► top 5 ──► LLM ──► answer
```

| Component | Choice |
|---|---|
| Parsing | BeautifulSoup + lxml, hidden XBRL blocks removed |
| Chunking | paragraph-packed, 1,500 chars, 200-char overlap, 1,263 chunks |
| Chunk header | `[Alphabet Google (GOOGL) 2026 \| Item 7 MD&A]` embedded with each chunk |
| Embeddings | `all-MiniLM-L6-v2`, normalized, FAISS `IndexFlatIP` (exact cosine) |
| Keyword search | BM25 (`rank-bm25`) |
| Fusion | Reciprocal Rank Fusion, k=60, candidate pool 50 per retriever |
| Generator | `openai/gpt-oss-120b` via Groq, temperature 0 |
| Judge | `qwen/qwen3.8-27b` via Groq: a **different model family** from the generator, to reduce self-preference bias |

---

## Evaluation set

30 hand-written questions: **6 per company**, **10 each** of three types:

- **numeric**: "What were AWS net sales in fiscal year 2025?"
- **factual**: "What are Microsoft's three reportable segments?"
- **risk**: "Which specific AI regulation does Microsoft cite as a risk to its European business?"

Each question has `gold_chunk_ids`: every chunk that answers it **on its own**.

**How gold labels were made.** Gold chunks were found with plain keyword search (`src/find_chunks.py`), a method independent of the retriever being tested. Labeling with the retriever's own output would make it grade its own homework: it could only ever score well on chunks it already finds.

Labeling rules:
- A chunk is gold only if it answers the question **by itself**. A table fragment containing `109,158` without the word "Services" doesn't count.
- For numeric questions, search the **number**, not the label. Labels vary ("Total revenue", "Revenue", "Total: Revenue"); the number doesn't. For Meta's revenue, label search found 4 gold chunks and number search found 9, including the consolidated income statement itself.
- Repeated passages count. 10-Ks copy text between sections, and chunk overlap duplicates text near boundaries, so one fact often lives in several chunks.
- Stratified by company, not by document size. Meta's risk section is 2.5× Apple's; proportional sampling would have made the scores mostly a measurement of Meta.

`src/validate_eval.py` checks the file for malformed rows, missing fields and duplicate IDs before any evaluation runs.

---

## Retrieval results

k = 5. **Recall@k** uses a capped denominator, `found / min(k, |gold|)`, so a question with 8 gold chunks can still reach 1.0.

| Mode | hit@5 | recall@5 | MRR |
|---|---|---|---|
| dense | 0.867 | 0.607 | 0.652 |
| **hybrid** | **0.933** | **0.632** | **0.679** |

By question type:

| Type | dense hit / recall / MRR | hybrid hit / recall / MRR |
|---|---|---|
| numeric | 0.80 / 0.297 / **0.560** | **1.00 / 0.397** / 0.453 |
| factual | 0.80 / 0.558 / 0.645 | 0.80 / **0.667 / 0.683** |
| risk | 1.00 / **0.967** / 0.750 | 1.00 / 0.833 / **0.900** |

**Hybrid is the default.** It wins on every overall metric and gets a correct chunk into the top 5 for every numeric question. Two costs remain and are real: on numeric questions it ranks the right chunk lower (MRR 0.453 vs 0.560), and on risk questions it retrieves fewer of the duplicate gold chunks (recall 0.833 vs 0.967). For compliance use, a missing number is worse than a less complete risk summary, so the trade favours hybrid.

Remaining misses: **q010** ("What does Apple manufacture and sell?") and **q022** (Alphabet's segments). Both are broad list questions, and both also fail in the generation eval: two independent metrics agreeing on the same failures.

---

## Generation results

Hybrid retrieval, k = 5.

| Type | correct | faithful | abstain | hallucination |
|---|---|---|---|---|
| factual | 0.80 | 1.00 | 0.10 | 0.00 |
| numeric | 1.00 | 1.00 | 0.00 | 0.00 |
| risk | 1.00 | 1.00 | 0.00 | 0.00 |
| **overall** | **0.933** | **1.00** | **0.033** | **0.00** |

Two separate judgments per answer, because they fail for different reasons:

- **correct**: does the answer match the reference answer? (numbers must match)
- **faithful**: is every claim supported by the retrieved chunks?

A wrong-but-faithful answer means **retrieval** failed; a wrong-and-unfaithful answer means the **model** invented something. Both failures here (q010, q022) are wrong-but-faithful: retrieval missed, and the model either answered narrowly from what it had or said "I don't know." It never filled the gap with invented content.

**Hallucination rate is computed over answered questions only.** Saying "I don't know" when the context lacks the answer is correct behaviour, not a failure.

**Judge verification.** Five verdicts were hand-checked on deliberately varied cases (an exact figure, a figure in billions, a two-part risk question, a judged failure, and the abstention): **5/5 agreement**. Five checks show the judge isn't broken; they don't prove it's perfect.

---

## What went wrong, and what it taught

These are the parts of the project that mattered most.

**1. The table of contents fooled the section parser.** In a 10-K, "Item 1A" appears in the table of contents, as the real heading, and in cross-references. A first fix that skipped the first 20% of each document worked for Microsoft and silently dropped Item 1A for the other four, because Apple's real heading sits at 12% of its document. The working rule filters on the **gap between matches**: TOC entries cluster within ~100 characters; real sections are thousands apart.

**2. The embedding model didn't know "Alphabet" means "GOOGL".** Chunk headers originally contained only the ticker. "What were Alphabet's advertising revenues?" returned **Meta** chunks, since Meta's MD&A says "advertising" constantly. Adding full company names to the embedded header fixed it. Metadata stored in a dict field is invisible to similarity search; if retrieval should use it, it has to be in the embedded text.

**3. Tables become number soup.** HTML-to-text extraction puts every table cell on its own line. The embedding model can find chunks that *discuss* revenue, but not the chunk that *contains* the figure. For "Alphabet's total revenues", dense retrieval returned the revenue-by-geography percentages and a highlights paragraph, the neighbours of the gold chunk, but not the gold chunk. On financial filings, much apparent hallucination is actually ingestion failure. BM25 matches the exact phrase "Total revenues" and fixes this case, which is why hybrid lifts numeric hit@5 from 0.80 to 1.00.

**4. A flawed measurement created a fake trade-off.** The first comparison showed hybrid *hurting* risk questions (hit@5 1.00 → 0.80). But the generation eval then showed four questions where retrieval scored zero gold chunks and the model still answered correctly. Checking the retrieved chunks showed all four were **labeling gaps**: three were chunk-overlap neighbours holding the same sentence as the labeled gold chunk, one was a second passage never found during labeling. After fixing the labels, hybrid ties dense on risk hit@5 and beats it overall. **The generation eval audited the retrieval eval.** When two metrics disagree, check the measurement before the system.

**5. Retry only what's retryable.** A retired model name returned `NotFoundError`, which a catch-everything retry loop spent 2.5 minutes retrying. The loop now retries only rate limits and connection errors and fails immediately on everything else.

---

## Limitations

- **Small eval set.** 30 questions, 10 per type: one question moves a per-type hit rate by 10 points. Differences of one or two questions are directional, not conclusive.
- **Lexical leakage.** Questions were written after reading their gold chunks and share their vocabulary, which likely inflates scores, especially for risk questions.
- **Single annotator.** All gold labels were made by one person; the audit found 4 gaps in 30 questions, so others may remain.
- **Labeling tool blind spots.** Keyword search misses text with irregular whitespace (`$ 214.4  billion`) and matches short terms inside other words ("AI" in "chairman").
- **LLM judge.** Hand-checked on 5 answers only. One open question: on q022 the judge says the segments were implied in the retrieved context while the model abstained; unresolved whether that is judge over-reading or model over-caution.
- **Table extraction.** Tables are flattened cell-by-cell; row-aware extraction would likely improve numeric recall further.
- **Scoped corpus.** Only Items 1, 1A, 7 and 8 are indexed.
- **Cost.** Runs on Groq's free tier; token counts are reported per query, dollar cost is not estimated.

---

## Reproduce

```bash
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key" > .env

python -m streamlit run app.py       # local web app (uses the prebuilt index)

python src/validate_eval.py          # check the eval set
python src/eval_retrieval.py dense
python src/eval_retrieval.py hybrid
python src/eval_generation.py hybrid
```

The prebuilt index in `faiss_store/` is committed, so the app and evals run without rebuilding. To rebuild the corpus and index from scratch:

```bash
python src/fetch_filings.py          # download 5 filings from EDGAR
python src/ingest.py                 # parse, chunk, embed, build index
```

Edit the `User-Agent` email in `fetch_filings.py` first; EDGAR rejects requests without real contact details. Note that rebuilding changes chunk IDs if the filings or chunking settings change, which would invalidate the gold labels in `eval/questions.jsonl`.

Results are written to `eval/results/`, with the retrieval mode and chunk size in each filename.

## Repository layout

```
app.py                 Streamlit web app
faiss_store/           prebuilt index (rebuild with src/ingest.py)
src/
  fetch_filings.py     EDGAR download
  parse.py             HTML to text, section extraction
  ingest.py            chunking, embeddings, FAISS index
  retriever.py         dense / bm25 / hybrid retrieval
  find_chunks.py       keyword search used to label gold chunks
  validate_eval.py     eval-set integrity checks
  eval_retrieval.py    hit@k, recall@k, MRR
  eval_generation.py   correctness, faithfulness, latency, tokens
  spot_check.py        print answers and judge verdicts for review
eval/
  questions.jsonl      30 questions with gold chunk IDs
  results/             saved evaluation runs
```
