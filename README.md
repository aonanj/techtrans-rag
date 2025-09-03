```markdown
# TechTrans — Technology Transactions RAG Assistant

Lightweight Flask application that provides a retrieval-augmented generation (RAG) interface for a corpus of technology-transaction documents. Upload documents, generate embeddings, and run queries that return both retrieved chunks and an optional model-generated answer.

## Summary

- Web UI: lightweight static pages served from `static/` (entry: `/`).
- API: document upload, embedding generation, and RAG query endpoints under `/api`.
- Storage: SQLite database for documents/chunks/embeddings (default `./data/app.db`) and filesystem folders for raw/clean text and metadata.

This README covers running the app locally, configuration, and the primary endpoints.

## Quick start (local)

1. Create a virtual environment and activate it (macOS / zsh):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install required packages. There is no pinned `requirements.txt` in the repo; the app expects typical packages such as `flask`, `python-dotenv`, `openai` (optional, for generation/embeddings), and `chromadb` (optional, for vector index). Example install:

```bash
pip install flask python-dotenv openai chromadb
```

3. Configure environment variables. Recommended (create a `.env` file):

```
# Optional: secret for Flask
SECRET_KEY=change-me

# OpenAI/LLM
OPENAI_API_KEY=<your-openai-key>
OPENAI_MODEL=gpt-5        # chat model default used when generating answers
EMBEDDING_MODEL=text-embedding-3-large

# Optional folders (defaults shown in config.py)
UPLOAD_FOLDER=./data/corpus_raw
CLEAN_FOLDER=./data/corpus_clean
METADATA_FOLDER=./data/metadata
INDEX_FOLDER=./data/index
DB_PATH=./data/app.db

# Controls
MAX_QUERY_LENGTH=1000
TOP_K_RESULTS=5
```

4. Run the app:

```bash
# Option A: run via the module entrypoint (recommended for quick dev)
python app.py

# Option B: use Flask CLI (factory pattern)
export FLASK_APP="app:create_app"
flask run --host 0.0.0.0 --port 8080
```

Open http://127.0.0.1:8080/ in your browser.

## Important files and folders

- `app.py` — application factory and CLI helper commands.
- `routes/api.py` — main API: `/api/upload`, `/api/embeddings/generate`, `/api/query`.
- `routes/web.py` — web routes for the UI (`/`, `/upload`).
- `infrastructure/` — utilities for chunking, embeddings, DB access, and document processing.
- `static/` — front-end pages (single-page UI in `index.html`).
- `data/` — default storage for DB, raw/clean text, metadata, and index (created automatically).

## API reference

All API endpoints are mounted under `/api`.

- POST `/api/upload`
	- Accepts multipart form with field name `file` (PDF/DOCX/TXT allowed).
	- Stores the raw file, extracts/cleans text, writes a manifest entry, saves a `doc_id` and chunks in the DB.
	- Response: `201` with JSON `{ "message": "Document uploaded successfully.", "filename": "<doc_id>" }` on success.

- POST `/api/embeddings/generate`
	- JSON body (optional): `{ "doc_id": "<doc_id>", "reindex": true|false }`.
	- Generates embeddings for any chunks missing vectors; if `reindex` true, optionally rebuilds Chroma index.
	- Response: JSON with generation/index counts and model used.

- POST `/api/query`
	- JSON body: `{ "query": "...", "top_k": 5, "docType": "...", "jurisdiction": "..." }`.
	- Returns nearest chunk matches and, if an OpenAI client/key is configured, a model-generated answer constrained to the retrieved context.
	- Response shape (successful):
	```json
	{
	"query": "...",
	"answer": "... or null",
	"matches": [ { "chunk_id", "doc_id", "doc_title", "doc_type", "section", "score", "preview" }, ... ],
	"used_model": "gpt-5 or null",
	"embedding_model": "text-embedding-3-large",
	"filters": { "docType": null, "jurisdiction": null },
	"timing_ms": { "retrieval": 123, "answer": 456 }
	}
	```

## CLI helpers (Flask commands)

Run these via the Flask CLI after setting `FLASK_APP="app:create_app"` or run `python app.py` to start the server directly.

- `flask list-docs` — prints all documents in the DB to stdout.
- `flask clear-db` — deletes all rows from `documents`, `chunks`, and `embeddings` tables.

## Storage & DB

- SQLite DB path configurable via `DB_PATH` (default `./data/app.db`). The DB schema is created automatically on app start by `infrastructure/database.py`.
- Manifest metadata is appended to `METADATA_FOLDER/manifest.jsonl` and includes `doc_type`, `jurisdiction`, and hashes.

## Notes for developers

- The project gracefully degrades when the OpenAI client or API key is missing: the `/api/query` endpoint will return an error about embedding generation being unavailable; embedding generation functions also log and return zero if OpenAI is not configured.
- `infrastructure/embeddings.py` stores vectors as float32 blobs in the `embeddings` table and optionally writes to a Chroma persistent collection if `chromadb` is installed and `INDEX_FOLDER` is configured.
- Chunking, title extraction, and manifest parsing happen during upload in `routes/api.py` using functions in `infrastructure/document_processor.py` and `infrastructure/chunker.py`.

## Environment variables (summary)

- `OPENAI_API_KEY` — API key used by `openai` client (required for embeddings & generation).
- `OPENAI_MODEL` / `CHAT_MODEL` — chat model used for answer generation (default `gpt-5` in code).
- `EMBEDDING_MODEL` — embedding model (default `text-embedding-3-large`).
- `UPLOAD_FOLDER`, `CLEAN_FOLDER`, `METADATA_FOLDER`, `INDEX_FOLDER`, `DB_PATH` — storage locations (see `config.py` for defaults).
- `MAX_QUERY_LENGTH`, `TOP_K_RESULTS` — request limits.

## Troubleshooting

- If uploads fail, check that the configured `UPLOAD_FOLDER` exists and is writable (the app attempts to create folders automatically).
- If `/api/query` returns an embedding generation error, ensure `OPENAI_API_KEY` is set and the `openai` client library (or `openai`-alias used in `infrastructure/embeddings.py`) is installed.

## Contributing

Contributions welcome. Please open issues or PRs describing bug fixes, improvements to chunking/embedding workflows, or tests.

## License

MIT License. Copyright (c) 2025 Alex S. Osterlind. See `LICENSE`.
