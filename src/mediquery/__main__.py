"""MediQuery CLI: ask questions over the synthetic clinical corpus."""

import argparse
import sys
from pathlib import Path

from .agents import answer_question
from .chunking import load_corpus
from .citations import format_sources
from .eval import run_eval
from .llm import OllamaClient
from .retrieval import BM25Index

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "synthetic"


def build_index(data_dir):
    index = BM25Index()
    index.add_documents(load_corpus(str(data_dir)))
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mediquery",
        description=(
            "MediQuery: agentic RAG over synthetic clinical documents. "
            "Demo only -- not medical advice."
        ),
    )
    parser.add_argument("question", nargs="?", help="Question to answer.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Corpus directory.")
    parser.add_argument("--top-k", type=int, default=4, help="Chunks to retrieve.")
    parser.add_argument("--eval", action="store_true", help="Run the eval harness.")
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model for generative answers (default: extractive, no LLM).",
    )
    args = parser.parse_args(argv)

    if args.eval:
        return 0 if run_eval(args.data, top_k=args.top_k) else 1

    if not args.question:
        parser.error("Provide a question or use --eval.")

    llm = None
    if args.model:
        llm = OllamaClient(model=args.model)
        if not llm.available():
            print(
                f"Ollama model '{args.model}' not reachable -- "
                "falling back to extractive answers.",
                file=sys.stderr,
            )
            llm = None

    index = build_index(args.data)
    result = answer_question(args.question, index, top_k=args.top_k, llm=llm)

    print(result["answer"])
    if result["sources"]:
        print("\nSources:")
        print(format_sources(result["sources"]))
    if result["issues"]:
        print("\nVerifier issues (answer not fully grounded):")
        for issue in result["issues"]:
            print(f"- {issue}")
    print("\nNote: demo over synthetic records only; not medical advice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
