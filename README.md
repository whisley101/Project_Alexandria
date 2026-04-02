# Project Alexandria

Local-first book RAG app built with Streamlit.

The app runs on your Mac, stores chunk embeddings in Pinecone, and uses Together AI to generate answers from retrieved book context.

## Architecture

- Streamlit UI runs locally.
- EPUB parsing, chunking, and embeddings happen locally in Python.
- Pinecone stores book vectors and metadata.
- Together AI generates the final answer from retrieved chunks.

## Requirements

- Python 3.10+ recommended
- A Pinecone account and API key
- A Together AI API key

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Create a local environment file:

```bash
cp .env.example .env
```

4. Fill in `.env` with your real values:

```env
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=project-alexandria
TOGETHER_API_KEY=...
```

## Run

Start the Streamlit app locally:

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

## One-Click Launcher

You can also launch the app from:

[`launch_app.command`](/Users/nathanrandall/Documents/Project_Alexandria/launch_app.command)

After dependencies are installed, double-clicking that file in Finder will:

- activate the local virtual environment
- start Streamlit
- open the app in your browser

## Notes

- The Pinecone index is created automatically if it does not already exist.
- Embeddings are generated locally with `BAAI/bge-large-en`, so the first ingestion step can be compute-heavy.
- Query embeddings are generated on demand and are not stored.
- Stored book chunk embeddings and metadata live in Pinecone.
