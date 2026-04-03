from __future__ import annotations

from typing import Callable, Dict, List, Optional

from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone

from embeddings import embedding_dimension
from utils import batched, require_env, slugify


BOOK_REGISTRY_NAMESPACE = "__books__"
UPSERT_BATCH_SIZE = 100
DEFAULT_PINECONE_CLOUD = "aws"
DEFAULT_PINECONE_REGION = "us-east-1"
BOOK_REGISTRY_SENTINEL_VALUE = 1e-12


def get_index():
    api_key = require_env("PINECONE_API_KEY")
    index_name = require_env("PINECONE_INDEX_NAME")
    client = Pinecone(api_key=api_key)
    ensure_index(client, index_name)
    return client.Index(index_name)


def ensure_index(client, index_name: str) -> None:
    if client.has_index(index_name):
        return

    client.create_index(
        name=index_name,
        dimension=embedding_dimension(),
        metric="cosine",
        spec=ServerlessSpec(
            cloud=DEFAULT_PINECONE_CLOUD,
            region=DEFAULT_PINECONE_REGION,
        ),
        deletion_protection="disabled",
    )


def _book_registry_vector(book_title: str) -> Dict:
    vector_dimension = embedding_dimension()
    return {
        "id": f"book::{slugify(book_title)}",
        # Pinecone rejects dense vectors that are entirely zero.
        "values": [BOOK_REGISTRY_SENTINEL_VALUE] * vector_dimension,
        "metadata": {
            "book": book_title,
            "kind": "book_registry",
        },
    }


def upsert_chunks(
    chunks: List[Dict],
    embeddings: List[List[float]],
    metadata: List[Dict],
    namespace: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    if not chunks:
        return

    index = get_index()
    vectors = []

    for chunk, embedding, chunk_metadata in zip(chunks, embeddings, metadata):
        vectors.append(
            {
                "id": f"{namespace}::chunk::{chunk['chunk_id']}",
                "values": embedding,
                "metadata": chunk_metadata,
            }
        )

    total_vectors = len(vectors)
    upserted = 0
    for batch in batched(vectors, UPSERT_BATCH_SIZE):
        index.upsert(vectors=list(batch), namespace=namespace)
        upserted += len(batch)
        if progress_callback:
            progress_callback(upserted, total_vectors)

    book_title = metadata[0]["book"]
    index.upsert(vectors=[_book_registry_vector(book_title)], namespace=BOOK_REGISTRY_NAMESPACE)


def query_chunks(
    query_embedding: List[float],
    book_filter: str,
    top_k: int = 5,
) -> List[Dict]:
    index = get_index()
    namespace = slugify(book_filter)
    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter={"book": {"$eq": book_filter}},
    )

    matches = getattr(response, "matches", None) or response.get("matches", [])
    results = []
    for match in matches:
        metadata = getattr(match, "metadata", None) or match.get("metadata", {})
        score = getattr(match, "score", None)
        if score is None:
            score = match.get("score")

        results.append(
            {
                "score": score,
                "book": metadata.get("book"),
                "chunk_id": metadata.get("chunk_id"),
                "text": metadata.get("text", ""),
                "page_estimate": metadata.get("page_estimate"),
            }
        )

    return results


def get_all_books() -> List[str]:
    index = get_index()
    ids: List[str] = []

    for page in index.list(namespace=BOOK_REGISTRY_NAMESPACE):
        ids.extend(page)

    if not ids:
        return []

    fetched = index.fetch(ids=ids, namespace=BOOK_REGISTRY_NAMESPACE)
    vectors = getattr(fetched, "vectors", None) or fetched.get("vectors", {})

    books = []
    for vector in vectors.values():
        metadata = getattr(vector, "metadata", None) or vector.get("metadata", {})
        book_title = metadata.get("book")
        if book_title:
            books.append(book_title)

    return sorted(set(books), key=str.lower)
