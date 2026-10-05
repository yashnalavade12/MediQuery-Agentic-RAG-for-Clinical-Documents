# MediQuery — Agentic RAG for Clinical Documents: Build Checklist

## v0.1 scaffold — DONE 2026-10-05 ✅
- [x] Repo scaffold: `data/synthetic/`, `src/mediquery/`, `eval/`, `tests/`, `README.md`
- [x] 6 synthetic clinical documents (SYN-001…SYN-006, fictional, no real PHI)
- [x] Section-aware chunking + BM25 retrieval with coverage re-rank and record-ID boost
- [x] Agent loop: planner → retriever → verifier → synthesizer (plain Python,
      each stage maps 1:1 to a future LangGraph node)
- [x] Extractive citation-grounded answers; Ollama client seam for Gemma 2B
- [x] 8-question eval harness: **8/8 passing** · 8 unit tests: **all passing**
- [x] Out-of-context abstention verified ("I don't have enough evidence…")
- [ ] Still roadmap (do NOT put on resume yet): PDF ingestion, dense
      embeddings + FAISS, LangGraph, clinical NER, 50-question eval,
      FastAPI, Streamlit UI, Docker

**Why this exists:** your finished one-page resume lists MediQuery as a built project.
Interviewers *will* go deep on an agentic RAG system. Make every resume bullet true
before you send that version out.

**Target:** done by ~Oct 25 (3 weeks). Stack: LangGraph, Ollama (Gemma 2B), FAISS,
FastAPI, Streamlit, Docker — all already in your toolbox.

## Day 1 — today (Sun Oct 4)
- [ ] Create the GitHub repo `MediQuery-Agentic-RAG-for-Clinical-Documents`
      (this exact name is already linked on your resume)
- [ ] Download the first batch of public clinical guideline PDFs
      (WHO guidelines, NICE, CDC — all public domain)
- [ ] Scaffold the repo: `ingest/`, `agents/`, `eval/`, `app/`, `Dockerfile`, `README.md`

## Week 1 (Oct 5–11): corpus + baseline RAG
- [ ] PDF ingestion: parsing + semantic chunking (~512 tokens with overlap)
- [ ] Local embeddings via Ollama (e.g. nomic-embed-text) → FAISS index over 100+ PDFs
- [ ] Baseline RAG working end-to-end: query → retrieve → grounded answer, fully offline
- [ ] Sanity check: 10 test questions, eyeball the answers

## Week 2 (Oct 12–18): agents + citations + NER + UI
- [ ] LangGraph pipeline: planner → retriever → verifier → synthesizer
- [ ] Verifier agent: cross-checks drafted claims against retrieved passages
      before final synthesis (your resume claims this — build it)
- [ ] Inline citations in answers ([1], [2] → source document + page)
- [ ] Clinical NER pipeline: extract conditions, medications, dosages
- [ ] Streamlit UI with a source/citation panel

## Week 3 (Oct 19–25): eval + ship
- [ ] Write the 50-question test set with expected citations
- [ ] Measure citation precision + faithfulness — record real numbers
      (if they beat the drafted resume numbers, update the resume)
- [ ] FastAPI backend; Dockerize the full stack
- [ ] README with architecture diagram + 2-min demo video
- [ ] ✅ Only now: send the resume version that lists MediQuery

## If you want to apply before Week 3
The honest options: (a) hold applications until the build is done, or
(b) send a version with MediQuery marked "In Progress" — ask Panda to make it.
Don't send the current version as-is; an unbuilt project is an interview trap.
