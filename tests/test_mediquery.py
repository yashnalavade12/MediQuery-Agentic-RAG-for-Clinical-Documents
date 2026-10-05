"""Unit tests for MediQuery (standard library unittest)."""

import unittest
from pathlib import Path

from mediquery.agents import answer_question, plan
from mediquery.chunking import load_corpus
from mediquery.retrieval import BM25Index

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"


def make_index():
    index = BM25Index()
    index.add_documents(load_corpus(str(DATA_DIR)))
    return index


class TestChunking(unittest.TestCase):
    def test_loads_six_documents(self):
        chunks = load_corpus(str(DATA_DIR))
        doc_ids = {c["doc_id"] for c in chunks}
        self.assertEqual(len(doc_ids), 6)

    def test_chunks_carry_metadata(self):
        for chunk in load_corpus(str(DATA_DIR)):
            for key in ("doc_id", "title", "section", "chunk_id", "text"):
                self.assertIn(key, chunk)
            self.assertTrue(chunk["text"].strip())


class TestRetrieval(unittest.TestCase):
    def test_xray_query_finds_radiology_report(self):
        index = make_index()
        results = index.search("What did the chest X-ray show?", top_k=3)
        self.assertTrue(results)
        self.assertEqual(results[0][0]["doc_id"], "SYN-002")

    def test_medication_query_finds_medication_list(self):
        index = make_index()
        results = index.search(
            "Which antibiotic was prescribed and for how long?", top_k=3
        )
        self.assertTrue(results)
        self.assertEqual(results[0][0]["doc_id"], "SYN-004")

    def test_unrelated_query_returns_nothing(self):
        index = make_index()
        self.assertEqual(index.search("What is the capital of France?", top_k=3), [])


class TestAgentLoop(unittest.TestCase):
    def test_answer_is_cited_and_verified(self):
        index = make_index()
        result = answer_question(
            "What antibiotic was prescribed for SYN-004 and for how long?", index
        )
        self.assertEqual(result["issues"], [])
        self.assertIn("[1]", result["answer"])
        self.assertTrue(result["sources"])
        self.assertIn("amoxicillin-clavulanate", result["answer"].lower())

    def test_out_of_context_abstains(self):
        index = make_index()
        result = answer_question("What is the capital of France?", index)
        self.assertEqual(result["sources"], [])
        self.assertIn("don't have enough evidence", result["answer"])

    def test_planner_splits_compound_questions(self):
        p = plan("What were the lab values on Sep 12 and what was the trend on Sep 14?")
        self.assertGreaterEqual(len(p["subqueries"]), 2)


if __name__ == "__main__":
    unittest.main()
