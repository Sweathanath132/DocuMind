"""Document loading + chunking.

Why chunk? An LLM has a limited context window and embeddings work best on
focused passages. We split documents into overlapping character windows so that
(a) each chunk is small enough to embed precisely, and (b) the overlap prevents
losing a sentence that straddles a boundary.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict
from .config import settings


def load_text(path: str | Path) -> str:
    """Extract raw text from .pdf or .txt/.md files."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    # plain text / markdown
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str,
               chunk_size: int | None = None,
               overlap: int | None = None) -> List[str]:
    """Sliding-window chunker that tries to break on paragraph/sentence
    boundaries instead of mid-word."""
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    text = " ".join(text.split())          # normalise whitespace
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # try to end on a sentence boundary for cleaner chunks
        if end < n:
            window = text[start:end]
            for sep in (". ", "? ", "! ", "; ", "\n"):
                cut = window.rfind(sep)
                if cut > chunk_size * 0.5:   # only if reasonably far in
                    end = start + cut + 1
                    break
        chunks.append(text[start:end].strip())
        start = max(end - overlap, end) if end == n else end - overlap
    return [c for c in chunks if c]


def build_chunks(paths: List[str | Path]) -> List[Dict]:
    """Return a list of {text, source, chunk_id} dicts ready for embedding."""
    records: List[Dict] = []
    for p in paths:
        text = load_text(p)
        for i, ch in enumerate(chunk_text(text)):
            records.append({"text": ch, "source": str(Path(p).name), "chunk_id": i})
    return records
