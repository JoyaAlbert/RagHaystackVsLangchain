LangChain + Chainlit RAG

This folder contains a LangChain + Chainlit replication of the Streamlit RAG app that exists in `haystack/`.

Files:
- `index_docs_langchain.py`: indexer that reads `doc/`, creates chunks and stores them in Chroma.
- `chainlit_app.py`: Chainlit app that handles user questions, retrieves context from Chroma and calls Google Gemini via `google.generativeai`.
- `utils.py`: helper functions to list/read files and chunk text (ported from `haystack/utils.py`).
- `requirements.txt`: minimal dependencies for this folder.

Usage (basic):
1. Create a virtualenv and install dependencies:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
2. Index documents:
   python index_docs_langchain.py
3. Run Chainlit (default port 8000):
   chainlit run chainlit_app.py

Set environment variables in a `.env` file (copy from repo root if exists):
- GOOGLE_API_KEY - required for Gemini calls
- CHROMA_PERSIST_DIRECTORY - optional (defaults to ./chroma_db)
- EMBEDDING_MODEL - optional (defaults to all-MiniLM-L6-v2)
