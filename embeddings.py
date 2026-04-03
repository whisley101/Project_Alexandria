from functools import lru_cache
from typing import Callable, List, Optional

from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-large-en"
EMBED_BATCH_SIZE = 16


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_texts(
    text_list: List[str],
    batch_size: int = EMBED_BATCH_SIZE,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[List[float]]:
    if not text_list:
        return []

    model = get_embedding_model()
    total = len(text_list)
    all_embeddings: List[List[float]] = []

    for start in range(0, total, batch_size):
        batch = text_list[start : start + batch_size]
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        all_embeddings.extend(embeddings.tolist())

        if progress_callback:
            progress_callback(min(start + len(batch), total), total)

    return all_embeddings


def embed_query(query: str) -> List[float]:
    query_embedding = embed_texts([query])
    return query_embedding[0] if query_embedding else []


def embedding_dimension() -> int:
    return get_embedding_model().get_sentence_embedding_dimension()
