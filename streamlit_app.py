"""Light-themed Streamlit chat UI for MediQuery."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mediquery.agents import answer_question
from mediquery.chunking import load_corpus
from mediquery.llm import OllamaClient
from mediquery.retrieval import BM25Index

MODEL = "gemma2:2b"
DATA_DIR = ROOT / "data" / "synthetic"
ABSTENTION = (
    "I don't have enough evidence in the provided documents to answer that reliably."
)

st.set_page_config(
    page_title="MediQuery",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f5f8fc;
    }
    [data-testid="stHeader"] {
        background: rgba(245, 248, 252, 0.9);
    }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5ebf3;
    }
    .block-container {
        max-width: 900px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    .mq-eyebrow {
        color: #3975c6;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .mq-subtitle {
        color: #526176;
        font-size: 1.05rem;
        margin-top: -0.6rem;
    }
    div[data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #e7edf5;
        border-radius: 16px;
        box-shadow: 0 4px 18px rgba(34, 59, 91, 0.04);
        padding: 1rem 1.15rem;
    }
    div[data-testid="stChatInput"] {
        background: #ffffff;
        border-color: #dce5f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def build_index(data_dir):
    index = BM25Index()
    index.add_documents(load_corpus(data_dir))
    return index


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources · {len(sources)} passages"):
        for number, source in enumerate(sources, start=1):
            st.markdown(
                f"**[{number}] {source['title']} — {source['section']}**  \n"
                f"`{source['doc_id']}` · `{source['chunk_id']}`"
            )
            st.write(source["text"])


with st.sidebar:
    st.markdown("## MediQuery")
    st.caption("Private, local question-answering over the demo corpus.")
    mode = st.radio(
        "Answer mode",
        ("Gemma 2B · Ollama", "Extractive · no LLM"),
        help="Gemma runs locally through Ollama. Extractive mode needs no model.",
    )
    st.divider()
    st.markdown("**Corpus**")
    st.caption("6 fictional clinical records · synthetic data only")
    st.warning(
        "Demo only. Not medical advice. Do not use real patient information.",
        icon="⚠️",
    )
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []

st.markdown('<p class="mq-eyebrow">Clinical document assistant</p>', unsafe_allow_html=True)
st.title("Ask your documents")
st.markdown(
    '<p class="mq-subtitle">Answers stay grounded in retrieved passages, '
    "with citations you can inspect.</p>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

index = build_index(str(DATA_DIR))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))
            if message.get("issues"):
                st.warning("Grounding check: " + "; ".join(message["issues"]))

question = st.chat_input("Ask about a record, medication, lab, or report…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    llm = None
    if mode == "Gemma 2B · Ollama":
        llm = OllamaClient(model=MODEL)
        if not llm.available():
            with st.chat_message("assistant"):
                st.error(
                    "Ollama is not responding at localhost:11434. Start Ollama "
                    "and try again, or choose Extractive mode."
                )
            st.stop()

    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching the records and checking citations…"):
                result = answer_question(question, index, llm=llm)
            content = result["answer"] or ABSTENTION
            st.markdown(content)
            render_sources(result["sources"])
            if result["issues"]:
                st.warning("Grounding check: " + "; ".join(result["issues"]))
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": content,
                "sources": result["sources"],
                "issues": result["issues"],
            }
        )
    except (OSError, ValueError) as exc:
        st.error(f"Could not generate an answer with Ollama: {exc}")
