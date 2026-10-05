"""Ingestion: read synthetic markdown clinical documents and chunk them."""

import re
from pathlib import Path


def _slug(text):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def chunk_markdown_file(path, max_chars=700):
    """Split one markdown file into section-aware chunks.

    Returns a list of dicts with doc_id, title, section, chunk_id, text.
    """
    text = Path(path).read_text(encoding="utf-8")
    name = Path(path).stem  # e.g. SYN-001-discharge-summary
    match = re.match(r"(SYN-\d+)", name)
    doc_id = match.group(1) if match else name

    title = None
    sections = []
    current_heading = "Overview"
    current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith(">"):
            # Skip the synthetic-data disclaimer blockquotes.
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading:
            if current_lines or sections or title is not None:
                sections.append((current_heading, current_lines))
            if len(heading.group(1)) == 1 and title is None:
                title = heading.group(2).strip()
                current_heading = "Overview"
                current_lines = []
            else:
                current_heading = heading.group(2).strip()
                current_lines = []
        else:
            current_lines.append(line)
    sections.append((current_heading, current_lines))

    if title is None:
        title = name.replace("-", " ")

    chunks = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        buffer = ""
        index = 0

        def flush():
            nonlocal buffer, index
            if buffer.strip():
                chunks.append(
                    {
                        "doc_id": doc_id,
                        "title": title,
                        "section": heading,
                        "chunk_id": f"{doc_id}#{_slug(heading)}#{index}",
                        "text": buffer.strip(),
                    }
                )
                index += 1
                buffer = ""

        for paragraph in paragraphs:
            if len(buffer) + len(paragraph) + 2 <= max_chars:
                buffer = (buffer + "\n\n" + paragraph).strip()
            else:
                flush()
                buffer = paragraph
        flush()

    return chunks


def load_corpus(data_dir):
    """Load and chunk every markdown document in a directory."""
    chunks = []
    for path in sorted(Path(data_dir).glob("*.md")):
        chunks.extend(chunk_markdown_file(path))
    return chunks
