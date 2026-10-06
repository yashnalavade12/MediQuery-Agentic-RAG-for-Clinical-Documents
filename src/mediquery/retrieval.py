"""BM25 retrieval over chunked clinical documents (standard library only)."""

import math
import re


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


# Filler words are stripped from *queries* so a question like
# "What is the capital of France?" doesn't match every chunk via "the".
# Document text is still indexed in full.
QUERY_STOPWORDS = frozenset(
    "what which when where who whom whose how does did was were is are the a an "
    "and or for of in on to with by as at it its this that these those do have "
    "has had be been".split()
)


class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self._docs = []  # list of (chunk, tokens)
        self._doc_lens = []
        self._df = {}
        self._avgdl = 0.0

    def add_documents(self, chunks):
        for chunk in chunks:
            tokens = tokenize(
                f"{chunk['title']} {chunk['section']} {chunk['text']}"
            )
            self._docs.append((chunk, tokens))
            self._doc_lens.append(len(tokens))
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1
        if self._docs:
            self._avgdl = sum(self._doc_lens) / len(self._docs)

    def _idf(self, term):
        df = self._df.get(term, 0)
        n = len(self._docs)
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def search(self, query, top_k=4):
        """Return [(chunk, score)] sorted by descending BM25 score."""
        query_terms = [t for t in tokenize(query) if t not in QUERY_STOPWORDS]
        scored = []
        for chunk, tokens in self._docs:
            tf = {}
            for term in tokens:
                tf[term] = tf.get(term, 0) + 1
            score = 0.0
            dl = len(tokens)
            for term in query_terms:
                if term not in tf:
                    continue
                denom = tf[term] + self.k1 * (
                    1 - self.b + self.b * dl / self._avgdl
                )
                score += self._idf(term) * tf[term] * (self.k1 + 1) / denom
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [(chunk, score) for score, chunk in scored[:top_k] if score > 0]