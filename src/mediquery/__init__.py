"""MediQuery — agentic RAG over synthetic clinical documents (demo).

Standard-library only. Not medical advice.
"""

from .agents import answer_question, plan
from .chunking import load_corpus
from .llm import OllamaClient
from .retrieval import BM25Index

__all__ = ["answer_question", "plan", "load_corpus", "BM25Index", "OllamaClient"]
