from pathlib import Path
from typing import Callable, Dict, List, Optional

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


def ingest_book(
    epub_path: str,
    book_title: str | None = None,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
) -> Dict:
    resolved_title = book_title or Path(epub_path).stem
    if progress_callback:
        progress_callback("extracting", 0, 1, "Extracting text from EPUB...")

    text = extract_text(epub_path)
    if progress_callback:
        progress_callback("chunking", 0, 1, "Chunking extracted text...")

    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No readable text was extracted from the EPUB.")

    if progress_callback:
        progress_callback("chunking", len(chunks), len(chunks), f"Created {len(chunks)} chunks.")

    chunk_texts = [chunk["text"] for chunk in chunks]
    if progress_callback:
        progress_callback("embedding", 0, len(chunks), f"Generating embeddings for 0/{len(chunks)} chunks...")

    embeddings = embed_texts(
        chunk_texts,
        progress_callback=lambda processed, total: progress_callback(
            "embedding",
            processed,
            total,
            f"Generating embeddings for {processed}/{total} chunks...",
        )
        if progress_callback
        else None,
    )
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

    if progress_callback:
        progress_callback("upserting", 0, len(chunks), f"Uploading 0/{len(chunks)} chunks to Pinecone...")

    upsert_chunks(
        chunks=chunks,
        embeddings=embeddings,
        metadata=metadata,
        namespace=namespace,
        progress_callback=lambda processed, total: progress_callback(
            "upserting",
            processed,
            total,
            f"Uploading {processed}/{total} chunks to Pinecone...",
        )
        if progress_callback
        else None,
    )

    if progress_callback:
        progress_callback("complete", len(chunks), len(chunks), f"Finished indexing {len(chunks)} chunks.")

    return {
        "book_title": resolved_title,
        "namespace": namespace,
        "chunk_count": len(chunks),
    }
