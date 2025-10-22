from typing import List
from PyPDF2 import PdfReader

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")  # keep going even if a page is weird
    return "\n\n".join(pages)

def chunk_text(
    text: str,
    chunk_size: int = 900,      # ~rough token proxy (characters)
    chunk_overlap: int = 200
) -> List[str]:
    """
    Simple char-based chunking to keep dependencies minimal.
    You can swap to token-aware chunking later if needed.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        # Try to end at a sentence boundary if possible
        sub = text[start:end]
        last_period = sub.rfind(". ")
        if last_period > 0 and end < n and (end - start) > 300:
            end = start + last_period + 2  # include ". "
            sub = text[start:end]
        chunks.append(sub)
        start = max(end - chunk_overlap, end)

    return [c.strip() for c in chunks if c.strip()]
