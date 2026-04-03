import time
from collections import deque
from typing import Callable, Deque, List, Optional, Tuple

from pinecone import Pinecone

from utils import require_env

MODEL_NAME = "llama-text-embed-v2"
EMBED_BATCH_SIZE = 24
EMBEDDING_DIMENSION = 1024
MAX_TOKENS_PER_MINUTE = 250_000
MAX_MODEL_TOKENS_PER_INPUT = 2048
TARGET_TOKENS_PER_MINUTE = 120_000
QUERY_INPUT_TYPE = "query"
PASSAGE_INPUT_TYPE = "passage"
MAX_EMBED_RETRIES = 5
RATE_LIMIT_SLEEP_SECONDS = 65


def get_inference_client() -> Pinecone:
    return Pinecone(api_key=require_env("PINECONE_API_KEY"))


def estimate_token_count(text: str) -> int:
    # Deliberately conservative approximation to avoid Pinecone TPM overruns.
    word_estimate = int(len(text.split()) * 1.8)
    char_estimate = max(1, len(text) // 3)
    return min(max(word_estimate, char_estimate, 1), MAX_MODEL_TOKENS_PER_INPUT)


def is_rate_limit_error(exc: Exception) -> bool:
    status = getattr(exc, "status", None)
    if status == 429:
        return True

    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "Too Many Requests" in text or "429" in text


def throttle_embed_requests(
    usage_window: Deque[Tuple[float, int]],
    requested_tokens: int,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    processed: int = 0,
    total: int = 0,
) -> None:
    while True:
        now = time.time()
        while usage_window and now - usage_window[0][0] >= 60:
            usage_window.popleft()

        used_tokens = sum(tokens for _, tokens in usage_window)
        if used_tokens + requested_tokens <= TARGET_TOKENS_PER_MINUTE:
            usage_window.append((now, requested_tokens))
            return

        wait_seconds = max(0.0, 60 - (now - usage_window[0][0]))
        if progress_callback and total:
            progress_callback(processed, total)
        time.sleep(wait_seconds + 0.1)


def embed_texts(
    text_list: List[str],
    batch_size: int = EMBED_BATCH_SIZE,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[List[float]]:
    if not text_list:
        return []

    client = get_inference_client()
    total = len(text_list)
    all_embeddings: List[List[float]] = []
    usage_window: Deque[Tuple[float, int]] = deque()

    for start in range(0, total, batch_size):
        batch = text_list[start : start + batch_size]
        requested_tokens = sum(estimate_token_count(text) for text in batch)
        throttle_embed_requests(
            usage_window,
            requested_tokens=min(requested_tokens, MAX_TOKENS_PER_MINUTE),
            progress_callback=progress_callback,
            processed=start,
            total=total,
        )
        response = None
        for attempt in range(MAX_EMBED_RETRIES):
            try:
                response = client.inference.embed(
                    model=MODEL_NAME,
                    inputs=batch,
                    parameters={"input_type": PASSAGE_INPUT_TYPE, "truncate": "END"},
                )
                break
            except Exception as exc:
                if not is_rate_limit_error(exc) or attempt == MAX_EMBED_RETRIES - 1:
                    raise

                # Reset the local usage window and wait out Pinecone's rolling minute cap.
                usage_window.clear()
                if progress_callback:
                    progress_callback(start, total)
                time.sleep(RATE_LIMIT_SLEEP_SECONDS)

        if response is None:
            raise ValueError("Pinecone embedding request did not return a response.")

        for item in response.data:
            values = item.get("values") if isinstance(item, dict) else getattr(item, "values", None)
            if values is None:
                raise ValueError("Pinecone embedding response did not include vector values.")
            all_embeddings.append(values)

        if progress_callback:
            progress_callback(min(start + len(batch), total), total)

    return all_embeddings


def embed_query(query: str) -> List[float]:
    client = get_inference_client()
    response = client.inference.embed(
        model=MODEL_NAME,
        inputs=[query],
        parameters={"input_type": QUERY_INPUT_TYPE, "truncate": "END"},
    )
    if not response.data:
        return []

    first_item = response.data[0]
    values = first_item.get("values") if isinstance(first_item, dict) else getattr(first_item, "values", None)
    if values is None:
        raise ValueError("Pinecone embedding response did not include query vector values.")
    return values


def embedding_dimension() -> int:
    return EMBEDDING_DIMENSION
