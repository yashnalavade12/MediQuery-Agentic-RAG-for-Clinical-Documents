"""Evaluation harness: grounded questions with expected citations.

Each question asserts three things:
  1. at least one expected document is cited,
  2. key facts appear in the answer,
  3. the verifier reports no issues.

v0.1 ships 8 questions over the synthetic corpus; the roadmap grows this
toward the 50-question clinical eval set.
"""

import json
from pathlib import Path

from .agents import answer_question
from .chunking import load_corpus
from .retrieval import BM25Index


def _default_questions_path(data_dir):
    return Path(data_dir).parent.parent / "eval" / "questions.json"


def run_eval(data_dir, questions_path=None, top_k=4):
    data_dir = Path(data_dir)
    qpath = Path(questions_path) if questions_path else _default_questions_path(data_dir)
    questions = json.loads(qpath.read_text(encoding="utf-8"))

    index = BM25Index()
    index.add_documents(load_corpus(str(data_dir)))

    passed = 0
    for q in questions:
        result = answer_question(q["question"], index, top_k=top_k)
        cited = {s["doc_id"] for s in result["sources"]}
        answer_low = result["answer"].lower()
        ok_doc = any(d in cited for d in q["expected_doc_ids"])
        ok_text = all(t.lower() in answer_low for t in q["must_contain"])
        ok = ok_doc and ok_text and not result["issues"]
        print(f"[{'PASS' if ok else 'FAIL'}] {q['id']}: {q['question']}")
        if not ok:
            print(f"  cited: {sorted(cited)} expected: {q['expected_doc_ids']}")
            print(f"  issues: {result['issues']}")
            print(f"  answer: {result['answer'][:220]}")
        passed += ok

    print(f"\n{passed}/{len(questions)} passed")
    return passed == len(questions)
