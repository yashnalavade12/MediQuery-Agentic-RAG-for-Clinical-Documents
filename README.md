# MediQuery — Agentic RAG for Clinical Documents

A privacy-first, offline question-answering system over clinical documents.
An agent loop — **planner → retriever → verifier → synthesizer** — answers
questions with inline citations, and **refuses to answer** when the retrieved
documents don't support one. In clinical RAG, a wrong confident answer is
worse than no answer.

> **Demo only.** The corpus is 100% synthetic, fictional data. This is not
> medical advice and must not be used for real clinical decisions.

## Architecture

```
  Question
     │
     ▼
 ┌─────────┐   sub-queries    ┌────────────┐   passages   ┌────────────┐
 │ Planner │ ───────────────▶ │ Retriever  │ ───────────▶ │ Verifier   │
 └─────────┘                  └────────────┘              └─────┬──────┘
        splits compound            BM25 + coverage        every sentence
        questions ("labs on        re-rank + record-ID     must carry a
        Sep 12 and trend on        boost                  valid citation;
        Sep 14" → 2 queries)                             fail → retry wider
                                                           │
                                                           ▼
                                                    ┌─────────────┐
                                                    │ Synthesizer │ ─▶ answer [1][2][3]
                                                    └─────────────┘   + source list
                                                     extractive (default)
                                                     or Ollama/Gemma 2B
```

**Why agentic, not plain RAG?** A baseline RAG pipeline retrieves then
generates and hopes. MediQuery's verifier checks the draft *before* it
reaches the user: every sentence must end with a citation (`… [2].`) that
points at a retrieved chunk topically relevant to the question. If
verification fails, retrieval broadens and the loop retries. If nothing
relevant is retrieved at all, the agent abstains:

> *I don't have enough evidence in the provided documents to answer that reliably.*

## What's implemented (v0.1, 2026-10-05)

- **Ingestion** (`chunking.py`): section-aware chunking of markdown clinical
  documents, with doc/section/chunk metadata per chunk.
- **Retrieval** (`retrieval.py`): BM25 with query stopword filtering, a
  coverage re-rank (chunks matching more distinct question terms win), and a
  record-ID boost (a question naming `SYN-002` prefers that record's chunks).
- **Agent loop** (`agents.py`): planner, retriever, verifier, synthesizer as
  plain functions — each maps 1:1 to a future LangGraph node, so the
  LangGraph migration is mechanical.
- **LLM seam** (`llm.py`): `OllamaClient` for local Gemma 2B generation
  (fully offline, no API keys). Falls back to deterministic extractive
  answers when Ollama isn't reachable, so the project runs anywhere on the
  standard library alone.
- **Eval harness** (`eval/`): 8 grounded questions asserting the right
  documents are cited, key facts appear, and the verifier is clean.
  **Current score: 8/8.**
- **Tests** (`tests/`): 8 unit tests (chunking, retrieval, abstention,
  citation verification). **All passing**, stdlib `unittest` only.
- **CLI** (`__main__.py`): ask questions, run evals, optionally use Ollama.
- **Streamlit UI** (`streamlit_app.py`): light-themed local chat interface with
  Gemma 2B via Ollama, extractive mode, and expandable cited passages.

## Roadmap (not yet built — see BUILD_CHECKLIST.md)

PDF ingestion of real guideline corpora (WHO/NICE/CDC) · dense embeddings
via Ollama + FAISS replacing BM25 · LangGraph orchestration · clinical NER
(conditions, medications, dosages) · 50-question eval set with citation
precision / faithfulness metrics · FastAPI backend · Docker.

## Run it

```bash
cd ~/workspace/mediquery

# Ask a question (extractive answers, no LLM needed)
PYTHONPATH=src python3 -m mediquery "What did the chest X-ray show for SYN-002?"

# Out-of-context → abstains instead of hallucinating
PYTHONPATH=src python3 -m mediquery "What is the chemotherapy protocol?"

# Run the eval harness
PYTHONPATH=src python3 -m mediquery --eval

# Run unit tests
PYTHONPATH=src python3 -m unittest discover -s tests

# Generative answers via local Ollama (optional)
PYTHONPATH=src python3 -m mediquery --model gemma2:2b "Which inhaler is listed for wheeze in SYN-004?"

# Light-themed chat UI (install requirements once; Ollama must be running for Gemma mode)
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

On Windows PowerShell, run these commands from the MediQuery project folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run .\streamlit_app.py
```

## Project layout

```
mediquery/
├── BUILD_CHECKLIST.md        # the full 3-week build plan
├── README.md                 # this file
├── streamlit_app.py          # light-themed chat UI
├── requirements.txt          # Streamlit UI dependency
├── .streamlit/config.toml    # light theme settings
├── data/synthetic/           # 6 fictional clinical documents (SYN-001…SYN-006)
├── eval/questions.json       # 8 grounded eval questions
├── src/mediquery/
│   ├── __init__.py
│   ├── __main__.py           # CLI
│   ├── agents.py             # planner → retriever → verifier → synthesizer
│   ├── chunking.py           # markdown ingestion + section-aware chunking
│   ├── citations.py          # source-list formatting
│   ├── eval.py               # eval harness
│   ├── llm.py                # Ollama client (local, offline)
│   └── retrieval.py          # BM25 + re-ranking
└── tests/test_mediquery.py   # unit tests
```

## Resume-safe wording (v0.1)

Until the roadmap items land, describe it honestly, e.g.:

> *Built an offline agentic RAG system over clinical documents with a
> planner → retriever → verifier → synthesizer loop; every answer is
> citation-grounded and the system abstains when evidence is insufficient.
> 8/8 on a grounded eval harness. Python, BM25 retrieval, standard library
> only — no cloud APIs, patient data never leaves the machine.*

Do **not** claim LangGraph, FAISS, NER, or a 50-question eval until they
exist — interviewers will probe an agentic RAG system deeply.
