from typing import Dict, List

from openai import OpenAI

from utils import require_env


DEFAULT_MODEL = "gpt-5.4-nano"
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

    api_key = require_env("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
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

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Answer strictly from the supplied context. If the answer is not in the context, say so clearly.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        ],
        max_output_tokens=700,
    )

    return response.output_text.strip()
