import os, sys, time, pathlib
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from retriever import Retriever

load_dotenv()
GEN_MODEL = "openai/gpt-oss-120b"
K = 5

PROMPT = """Answer the question using ONLY the context below from SEC 10-K filings.
If the context does not contain the answer, reply exactly: I don't know.
Give figures with their units. Be concise.

Context:
{context}

Question: {question}"""


@st.cache_resource                      # load model + index once, not on every click
def load():
    return Retriever(mode="hybrid"), Groq(api_key=os.environ["GROQ_API_KEY"])


st.set_page_config(page_title="Ask My Notes: 10-K RAG", layout="wide")
st.title("Ask My Notes: 10-K RAG")
st.caption("Hybrid retrieval over the latest 10-K filings of Apple, Microsoft, Alphabet, "
           "Amazon and Meta. Evaluated: 93% answer correctness, 0% hallucination on a "
           "30-question gold set.")

examples = ["What were AWS net sales in fiscal year 2025?",
            "Which specific AI regulation does Microsoft cite as a risk to its European business?",
            "What are Meta's reportable segments?"]
question = st.text_input("Ask a question about the filings", placeholder=examples[0])
st.markdown("**Try:** " + " · ".join(f"_{e}_" for e in examples))

if question:
    retriever, client = load()

    t0 = time.perf_counter()
    chunks = retriever.retrieve(question, k=K)
    t_ret = time.perf_counter() - t0

    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=GEN_MODEL, temperature=0,
        messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}])
    t_gen = time.perf_counter() - t0

    st.subheader("Answer")
    st.markdown(resp.choices[0].message.content)
    st.caption(f"Retrieval {t_ret:.2f}s · Generation {t_gen:.2f}s · "
               f"{resp.usage.prompt_tokens} prompt / {resp.usage.completion_tokens} completion tokens")

    st.subheader("Sources")
    for c in chunks:
        header, _, body = c["text"].partition("\n")
        with st.expander(f"#{c['chunk_id']}  {header}"):
            st.text(body[:1500])
            