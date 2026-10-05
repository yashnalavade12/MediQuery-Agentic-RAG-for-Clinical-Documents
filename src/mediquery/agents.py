"""Agentic RAG loop: planner -> retriever -> verifier -> synthesizer.

Framework-free on purpose so it runs anywhere with the standard library.
Each stage mirrors a future LangGraph node, so migrating to LangGraph later
is mechanical: one node per function below.
"""

import re

from .retrieval import QUERY_STOPWORDS, tokenize

STOPWORDS = frozenset(
    "what which when where who whom whose how does did was were is are the a an "
    "and or for of in on to with by as at it its this that these those between "
    "from show shown give given list listed".split()
)


def significant_terms(text):
    """Content-bearing tokens used for extraction scoring and verification."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


def _chunk_context(chunk):
    return f"{chunk['title']} {chunk['section']} {chunk['text']}".lower()


def plan(question):
    """Planner: split a question into focused sub-queries.

    A compound question ("labs on Sep 12 and trend on Sep 14") becomes two
    retrieval steps instead of one muddy query.
    """
    parts = re.split(r"\s+and\s+|;\s*|\?\s*", question)
    subqueries = []
    for part in parts:
        part = part.strip(" ?.,")
        if part and part.lower() not in {s.lower() for s in subqueries}:
            subqueries.append(part)
    if not subqueries:
        subqueries = [question.strip()]
    return {"subqueries": subqueries[:3]}


def retrieve(plan_obj, index, top_k=4):
    """Retriever: run each sub-query, merge, de-duplicate, and re-rank.

    Re-ranking prefers chunks covering more distinct question terms
    (lexical recall) with BM25 as the tiebreak. This keeps small stub
    chunks (e.g. a patient header matching only the ID) from outranking
    the content chunk that actually answers the question.
    """
    seen = set()
    candidates = []
    # Pull a wide pool per sub-query; the re-ranker below decides the final
    # top_k. A narrow per-query cut would drop the answering chunk before
    # re-ranking ever sees it.
    per_query = max(top_k * 2, 8)
    for subquery in plan_obj["subqueries"]:
        for chunk, score in index.search(subquery, top_k=per_query):
            if chunk["chunk_id"] not in seen:
                seen.add(chunk["chunk_id"])
                candidates.append((chunk, score))

    all_terms = set()
    wanted_ids = set()
    for subquery in plan_obj["subqueries"]:
        all_terms.update(t for t in tokenize(subquery) if t not in QUERY_STOPWORDS)
        wanted_ids.update(re.findall(r"syn-\d+", subquery.lower()))

    def coverage(chunk):
        # Token-set matching (not substring): "antibiotics" must not count
        # as "antibiotic", or wrong-patient chunks slip in.
        tokens = set(
            tokenize(f"{chunk['title']} {chunk['section']} {chunk['text']}")
        )
        return sum(1 for term in all_terms if term in tokens)

    def sort_key(item):
        chunk, score = item
        # An explicitly named record ID is strong intent: its chunks rank first.
        id_match = 1 if chunk["doc_id"].lower() in wanted_ids else 0
        return (id_match, coverage(chunk), score)

    candidates.sort(key=sort_key, reverse=True)
    return [chunk for chunk, _ in candidates[:top_k]]


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _clean_text(text):
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            continue
        stripped = re.sub(r"^(\d+\.\s+|[-*]\s+)", "", stripped)
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


def _cite(sentence, i):
    """Attach a citation *inside* the sentence, before terminal punctuation.

    "No pleural effusion." -> "No pleural effusion [1]." so sentence
    splitters keep the citation with the claim it supports.
    """
    match = re.match(r"^(.*)([.!?])$", sentence, re.DOTALL)
    if match:
        return f"{match.group(1)} [{i}]{match.group(2)}"
    return f"{sentence} [{i}]"


def extract_relevant_sentences(chunk_text, query_terms, max_sentences=2):
    """Pick the sentences in a chunk that best match the question terms.

    Token-overlap scoring (not substring): otherwise 'syn' would match
    inside 'synthetic' and the wrong sentence would win.
    """
    sentences = split_sentences(_clean_text(chunk_text))
    term_set = set(query_terms)
    scored = []
    for sentence in sentences:
        hits = len(set(tokenize(sentence)) & term_set)
        scored.append((hits, sentence))
    scored.sort(key=lambda item: item[0], reverse=True)
    picked = [s for hits, s in scored if hits > 0][:max_sentences]
    if not picked and sentences:
        picked = [sentences[0]]
    return " ".join(picked)


def synthesize(question, chunks, query_terms, llm=None):
    """Synthesizer: draft a citation-grounded answer.

    Without an LLM this is extractive -- every sentence is quoted verbatim
    from a cited chunk, so nothing can be hallucinated. With an LLM (Ollama),
    the model drafts from the retrieved passages and the verifier still
    checks every citation.
    """
    if llm is None:
        sentences = []
        sources = []
        for i, chunk in enumerate(chunks[:3], start=1):
            excerpt = extract_relevant_sentences(chunk["text"], query_terms)
            # Every sentence carries its own citation so the verifier
            # can check each claim independently.
            for sentence in split_sentences(excerpt):
                sentences.append(_cite(sentence, i))
            sources.append(chunk)
        return " ".join(sentences), sources

    context_blocks = []
    for i, chunk in enumerate(chunks[:4], start=1):
        context_blocks.append(
            f"[{i}] {chunk['title']} — {chunk['section']}:\n{chunk['text']}"
        )
    prompt = (
        "Answer the question using ONLY the passages below. "
        "Every sentence of your answer must include its citation before the "
        "final period, like this: 'The X-ray showed an infiltrate [1].' "
        "If the passages do not contain the answer, say you do not have "
        "enough evidence.\n\n"
        f"Passages:\n{chr(10).join(context_blocks)}\n\n"
        f"Question: {question}\nAnswer:"
    )
    return llm.generate(prompt).strip(), chunks[:4]


def verify(answer, sources, query_terms):
    """Verifier: every sentence must carry a valid, on-topic citation.

    v0.1 checks citation validity (exists, in range) and that the cited
    chunk is topically relevant to the question. Claim-level entailment
    checking is on the roadmap.
    """
    issues = []
    for sentence in split_sentences(answer):
        cited = [int(n) - 1 for n in re.findall(r"\[(\d+)\]", sentence)]
        if not cited:
            issues.append(f"Sentence lacks citation: {sentence[:70]}")
            continue
        for idx in cited:
            if idx < 0 or idx >= len(sources):
                issues.append(f"Citation [{idx + 1}] is out of range.")
                continue
            if not any(term in _chunk_context(sources[idx]) for term in query_terms):
                issues.append(
                    f"Citation [{idx + 1}] does not match the question topic."
                )
    return issues


def answer_question(question, index, top_k=4, max_retries=2, llm=None):
    """Run the full agent loop: plan -> retrieve -> synthesize -> verify.

    If verification fails, retrieval is broadened and the loop retries.
    If nothing relevant is retrieved at all, the agent abstains instead of
    guessing -- refusing is a feature in clinical RAG.
    """
    terms = significant_terms(question)
    plan_obj = plan(question)
    attempt = 0
    k = top_k
    while True:
        chunks = retrieve(plan_obj, index, top_k=k)
        if not chunks:
            return {
                "answer": (
                    "I don't have enough evidence in the provided documents "
                    "to answer that reliably."
                ),
                "sources": [],
                "issues": [],
                "plan": plan_obj,
            }
        answer, sources = synthesize(question, chunks, terms, llm=llm)
        issues = verify(answer, sources, terms)
        if not issues or attempt >= max_retries:
            return {
                "answer": answer,
                "sources": sources,
                "issues": issues,
                "plan": plan_obj,
            }
        attempt += 1
        k += 2
