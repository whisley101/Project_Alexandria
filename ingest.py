from pathlib import Path
from typing import Dict, List

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub

from embeddings import embed_texts
from utils import compact_whitespace, slugify
from vector_store import upsert_chunks


WORDS_PER_CHUNK = 800
OVERLAP_WORDS = 100
WORDS_PER_PAGE_ESTIMATE = 300


def extract_text(epub_path: str) -> str:
    book = epub.read_epub(epub_path)
    sections: List[str] = []

    for item in book.get_items():
        if item.get_type() != ITEM_DOCUMENT:
            continue

        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        cleaned = compact_whitespace(text)
        if cleaned:
            sections.append(cleaned)

    return "\n\n".join(sections)


def chunk_text(text: str, chunk_size: int = WORDS_PER_CHUNK, overlap: int = OVERLAP_WORDS) -> List[Dict]:
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - overlap)
    chunks: List[Dict] = []

    for chunk_id, start in enumerate(range(0, len(words), step)):
        end = min(len(words), start + chunk_size)
        chunk_words = words[start:end]
        if not chunk_words:
            continue

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": " ".join(chunk_words),
                "word_start": start,
                "word_end": end,
                "page_estimate": (start // WORDS_PER_PAGE_ESTIMATE) + 1,
            }
        )

        if end >= len(words):
            break

    return chunks


def ingest_book(epub_path: str, book_title: str | None = None) -> Dict:
    resolved_title = book_title or Path(epub_path).stem
    text = extract_text(epub_path)
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No readable text was extracted from the EPUB.")

    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(chunk_texts)
    namespace = slugify(resolved_title)

    metadata = [
        {
            "book": resolved_title,
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "page_estimate": chunk["page_estimate"],
        }
        for chunk in chunks
    ]

    upsert_chunks(chunks=chunks, embeddings=embeddings, metadata=metadata, namespace=namespace)

    return {
        "book_title": resolved_title,
        "namespace": namespace,
        "chunk_count": len(chunks),
    }
