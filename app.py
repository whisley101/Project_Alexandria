from __future__ import annotations

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from embeddings import embed_query
from ingest import ingest_book
from llm import generate_answer
from utils import truncate
from vector_store import get_all_books, query_chunks


load_dotenv()
st.set_page_config(page_title="Book Query App", page_icon="📚", layout="wide")


def render_env_help() -> None:
    missing = [
        name
        for name in ["PINECONE_API_KEY", "TOGETHER_API_KEY", "PINECONE_INDEX_NAME"]
        if not os.getenv(name)
    ]
    if missing:
        st.warning(
            "Missing environment variables: "
            + ", ".join(missing)
            + ". Add them before using ingestion or queries."
        )


def main() -> None:
    st.title("📚 Book Query App")
    st.caption("Upload EPUBs, index them in Pinecone, and ask focused questions about a selected book.")
    render_env_help()

    left_col, right_col = st.columns([1, 1.2], gap="large")

    with left_col:
        st.subheader("Ingest Book")
        uploaded_file = st.file_uploader("Upload an EPUB", type=["epub"])
        book_title = st.text_input("Book title override (optional)")

        if st.button("Process Book", use_container_width=True):
            if not uploaded_file:
                st.error("Please upload an EPUB file first.")
            else:
                suffix = os.path.splitext(uploaded_file.name)[1] or ".epub"
                temp_path = None
                progress_bar = st.progress(0)
                progress_text = st.empty()

                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                        temp_path = temp_file.name

                    def update_ingestion_progress(stage: str, processed: int, total: int, message: str) -> None:
                        if total <= 0:
                            progress_bar.progress(0)
                        else:
                            progress_bar.progress(min(processed / total, 1.0))
                        progress_text.caption(message)

                    result = ingest_book(
                        temp_path,
                        book_title=book_title or uploaded_file.name.rsplit(".", 1)[0],
                        progress_callback=update_ingestion_progress,
                    )

                    progress_bar.progress(1.0)
                    progress_text.caption(f"Finished indexing {result['chunk_count']} chunks.")
                    st.success(
                        f"Indexed '{result['book_title']}' with {result['chunk_count']} chunks."
                    )
                except Exception as exc:
                    st.error(f"Book ingestion failed: {exc}")
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)

    with right_col:
        st.subheader("Ask Questions")
        try:
            books = get_all_books()
        except Exception as exc:
            books = []
            st.error(f"Could not load books from Pinecone: {exc}")

        if not books:
            st.info("No indexed books found yet. Upload and process a book to get started.")
            return

        selected_book = st.selectbox("Select a book", options=books)
        user_query = st.text_input("Ask a question about the selected book")
        top_k = st.slider("Number of chunks to retrieve", min_value=3, max_value=10, value=5)

        if st.button("Ask", use_container_width=True):
            if not user_query.strip():
                st.error("Enter a question before querying.")
            else:
                try:
                    with st.spinner("Searching relevant sections..."):
                        query_embedding = embed_query(user_query)
                        matches = query_chunks(query_embedding, selected_book, top_k=top_k)

                    if not matches:
                        st.warning("No matching sections were found for that book.")
                        return

                    with st.spinner("Generating answer..."):
                        answer = generate_answer(user_query, matches)

                    st.subheader("Answer")
                    st.write(answer)

                    st.subheader("Top Matches")
                    for idx, match in enumerate(matches, start=1):
                        with st.container(border=True):
                            st.markdown(
                                f"**Match {idx}**  \n"
                                f"Book: `{match['book']}`  \n"
                                f"Chunk: `{match['chunk_id']}`  \n"
                                f"Page estimate: `{match['page_estimate']}`"
                            )
                            st.write(truncate(match["text"], 300))
                except Exception as exc:
                    st.error(f"Query failed: {exc}")


if __name__ == "__main__":
    main()
