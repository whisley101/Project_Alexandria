from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-large-en"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_texts(text_list: List[str]) -> List[List[float]]:
    if not text_list:
        return []

    model = get_embedding_model()
    embeddings = model.encode(
        text_list,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    query_embedding = embed_texts([query])
    return query_embedding[0] if query_embedding else []


def embedding_dimension() -> int:
    return get_embedding_model().get_sentence_embedding_dimension()
