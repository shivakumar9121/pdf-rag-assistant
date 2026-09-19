# PDF RAG Assistant

Ask questions across uploaded PDFs using hybrid retrieval and a streamed language-model response with filename/page citation prompts.

Built by **Pathlavath Shiva Kumar**. [Portfolio](https://shivakumar9121.github.io/pdf-rag-assistant.html) · [Contact](mailto:pathlavathshivakumar978@gmail.com)

## What it implements

- Multiple PDF uploads and a chat interface in Streamlit.
- Page-aware text extraction with PyMuPDF and overlapping character chunks.
- Normalized BGE-small embeddings indexed in ChromaDB.
- Hybrid candidate retrieval: BM25 keyword ranking plus vector similarity search.
- Deduplication and BGE cross-encoder reranking before selecting context.
- Streamed responses through Groq's OpenAI-compatible API, with instructions to cite the source filename and page.

## Architecture

```text
PDF uploads → PyMuPDF extraction → overlapping chunks + page metadata
                                      ↓
                             BGE embeddings → ChromaDB
                                      ↓
Question → BM25 + vector candidates → deduplicate → cross-encoder reranking
                                      ↓
                         top passages → Groq → streamed answer
```

The Streamlit app uses an ephemeral Chroma client. The separate ingestion entry point uses a persistent local Chroma directory.

## Stack

Python 3.11+, Streamlit, PyMuPDF, ChromaDB, Sentence Transformers, `rank-bm25`, and the OpenAI Python client configured for Groq.

- Embedding model: `BAAI/bge-small-en-v1.5`
- Reranker: `BAAI/bge-reranker-base`
- Current answer model: `openai/gpt-oss-120b` through Groq
- Default chunks: 800 characters with 100-character overlap
- Retrieval: up to 10 candidates from each method, reranked to 5 passages

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use Python 3.11 or newer.

```bash
git clone https://github.com/shivakumar9121/pdf-rag-assistant.git
cd pdf-rag-assistant
uv sync
```

Create a local `.env` file using `.env.example` as the template and set your own `GROQ_API_KEY`. Keep it out of version control. Then launch:

```bash
uv run --env-file .env streamlit run notes_gpt/app.py
```

The explicit `--env-file` loads the key for code that reads environment variables. The first run downloads the embedding and reranking models. Answer generation requires network access and a valid Groq API key.

## Source map

- `notes_gpt/app.py` — uploads, session state and Streamlit chat interface
- `notes_gpt/ingest.py` — PDF extraction, chunking, embedding and indexing
- `notes_gpt/retrieve.py` — BM25/vector retrieval and cross-encoder reranking
- `notes_gpt/chat.py` — context prompt, Groq client and response streaming
- `evals/` — a separate experimental evaluation pipeline and recorded outputs

## Evaluation and current limits

The recorded evaluation files use a separate LlamaIndex pipeline, different from the main hybrid-retrieval application. Their scores are historical experiment outputs, not validated metrics for the current app. The source evaluation PDFs and all optional evaluation dependencies are not included in this repository.

- Image-only PDFs require an OCR stage, which is not implemented.
- Citation and context-only instructions do not guarantee factual answers; verify important answers against the source pages.
- The retriever has no relevance rejection threshold for weak matches.
- The clear-documents control resets application references but does not explicitly delete the Chroma collection.
- Retrieved passages are sent to Groq for answer generation. Only upload documents you are authorized to process through that service.

Useful next improvements include reproducible retrieval tests, a fixed evaluation document set, relevance thresholds and explicit collection cleanup.


