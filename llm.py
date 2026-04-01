from typing import Dict, List

from together import Together

from utils import require_env, truncate


DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
MAX_CONTEXT_CHARS = 12000


def _format_contexts(contexts: List[Dict]) -> str:
    sections = []
    total_chars = 0

    for item in contexts:
        section = (
            f"Book: {item.get('book')}\n"
            f"Chunk: {item.get('chunk_id')}\n"
            f"Page estimate: {item.get('page_estimate')}\n"
            f"Excerpt: {item.get('text')}\n"
        )
        total_chars += len(section)
        if total_chars > MAX_CONTEXT_CHARS:
            break
        sections.append(section)

    return "\n---\n".join(sections)


def generate_answer(query: str, contexts: List[Dict], model: str = DEFAULT_MODEL) -> str:
    if not contexts:
        return "No relevant sections were found for that question."

    api_key = require_env("TOGETHER_API_KEY")
    client = Together(api_key=api_key)
    prompt = (
        "You are answering questions about a book.\n\n"
        "Use ONLY the context below.\n\n"
        "Return:\n"
        "- Summary\n"
        "- Key supporting excerpts\n"
        "- Page or location references\n\n"
        f"Context:\n{_format_contexts(contexts)}\n\n"
        f"Question:\n{query}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Answer strictly from the supplied context. If the answer is not in the context, say so clearly.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=700,
    )

    return response.choices[0].message.content.strip()
