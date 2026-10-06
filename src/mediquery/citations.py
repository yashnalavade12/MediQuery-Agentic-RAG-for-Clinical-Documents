"""Citation formatting helpers."""


def format_sources(sources):
    """Render the numbered source list for an answer."""
    lines = []
    for i, source in enumerate(sources, start=1):
        lines.append(
            f"[{i}] {source['title']} — {source['section']} "
            f"({source['doc_id']}, {source['chunk_id']})"
        )
    return "\n".join(lines)