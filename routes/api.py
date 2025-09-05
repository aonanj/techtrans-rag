# api.py (Cloud Run + PostgreSQL refactor)
import os
import json
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from infrastructure.logger import get_logger

# Swap old infrastructure.database for the local database.py refactor
# Provides SQLAlchemy sessions and helpers for Cloud SQL Postgres
from infrastructure.database import (
    init_db,
    add_document,
    add_chunk,
    get_document_by_sha,
    get_chunks_for_doc,
)
from infrastructure.embeddings import generate_and_store_embeddings, find_nearest_neighbors_advanced
# Keep existing processing utilities
from infrastructure.document_processor import (
    extract_text,
    extract_title,
    get_manifest_info,
    sha256_text,
)
from infrastructure.chunker import chunk_doc
# If you have a new retrieval path (Vertex AI Vector Search), wire it in here:
from infrastructure.vector_search import find_nearest_neighbors
from infrastructure.database import get_chunks_by_ids

api_bp = Blueprint("api", __name__, url_prefix="/api")
logger = get_logger()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _upload_dirs():
    base = os.getenv("UPLOAD_FOLDER", os.path.join(os.getcwd(), ".data/uploads"))
    raw_dir = os.path.join(base, "raw")
    clean_dir = os.path.join(base, "clean")
    meta_dir = os.path.join(base, "metadata")
    ensure_dir(raw_dir)
    ensure_dir(clean_dir)
    ensure_dir(meta_dir)
    return raw_dir, clean_dir, meta_dir

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@api_bp.route("/upload", methods=["POST"])
def add_doc():
    """Upload one document, extract text, persist metadata in Postgres, and schedule chunking.

    Optional form fields:
      doc_type: str
      jurisdiction: str
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    # Optional metadata fields (multipart/form-data text inputs)
    doc_type = request.form.get("doc_type") or None
    jurisdiction = request.form.get("jurisdiction") or None

    logger.info("Received upload: filename=%s, doc_type=%s, jurisdiction=%s",
                file.filename, doc_type, jurisdiction)

    raw_dir, clean_dir, meta_dir = _upload_dirs()
    logger.info("Upload directories: raw=%s, clean=%s, meta=%s", raw_dir, clean_dir, meta_dir)
    filename = secure_filename(file.filename)
    logger.info("Secured filename: %s", filename)
    raw_path = os.path.join(raw_dir, filename)
    logger.info("Saving uploaded file to %s", raw_path)
    file.save(raw_path)
    logger.info("File uploaded to %s", raw_path)

    # Extract text and title
    try:
        text = extract_text(raw_path)
        title = extract_title(text) or filename
    except Exception as e:
        logger.exception("Text extraction failed: %s", e)
        return jsonify({"error": "Failed to extract text"}), 500

    # Persist clean text to disk for transparency (optional)
    clean_name = f"{os.path.splitext(filename)[0]}.txt"
    clean_path = os.path.join(clean_dir, clean_name)
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Compute content hash and upsert document row
    content_sha = sha256_text(text)
    doc = get_document_by_sha(content_sha)
    if not doc:
        # Insert new
        add_document(sha256=content_sha, title=title, source_path=raw_path, doc_type=doc_type, jurisdiction=jurisdiction)
        doc = get_document_by_sha(content_sha)
    else:
        # Enrich existing if metadata blank
        try:
            add_document(sha256=content_sha, doc_type=doc_type, jurisdiction=jurisdiction)
            doc = get_document_by_sha(content_sha)
        except Exception:
            pass

    if not doc:
        logger.error("Failed to create or retrieve document for SHA: %s", content_sha)
        return jsonify({"error": "Failed to process document in database"}), 500

    # Ensure we have a concrete integer primary key (avoid Column / class attribute issues)
    doc_id_val = getattr(doc, "doc_id", None)
    if not isinstance(doc_id_val, int):
        logger.error("Document record missing integer doc_id: %s", doc)
        return jsonify({"error": "Invalid document id"}), 500

    # Write manifest.jsonl record to local metadata dir for compatibility
    manifest = get_manifest_info(text)
    manifest_record = {
        "doc_id": doc_id_val,
        "title": title,
        "sha256": content_sha,
        "source_path": raw_path,
        "clean_path": clean_path,
        "bytes": os.path.getsize(raw_path),
        "content_type": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest,
    }
    ensure_dir(os.path.join(meta_dir, str(doc_id_val)))
    manifest_file = os.path.join(meta_dir, str(doc_id_val), "manifest.jsonl")
    with open(manifest_file, "a", encoding="utf-8") as mf:
        mf.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")

    # Chunk now. Persist chunks into Postgres via add_chunk.
    try:
        chunks = chunk_doc(file_path=clean_path, doc_id=str(doc_id_val))
        # Expected `chunks` is iterable of dicts: {text, chunk_index, token_count?}
        persisted = 0
        for ch in chunks or []:
            raw_text = ch.get("text") if isinstance(ch, dict) else ch
            if raw_text is None:
                # Skip chunks without text to satisfy type requirements
                continue
            text_val = str(raw_text)
            idx = ch.get("chunk_index", persisted) if isinstance(ch, dict) else persisted
            tok = ch.get("token_count") if isinstance(ch, dict) else None
            add_chunk(doc_id=doc_id_val, text=text_val, chunk_index=idx, token_count=tok)
            persisted += 1
    except Exception as e:
        logger.exception("Chunking failed: %s", e)
        # Do not fail the upload if chunking fails. The worker can retry.
        pass

    return jsonify({
        "message": "ok",
        "doc_id": doc_id_val,
        "sha256": content_sha,
        "title": title,
        "doc_type": getattr(doc, "doc_type", None),
        "jurisdiction": getattr(doc, "jurisdiction", None)
    }), 200


@api_bp.route("/chunks", methods=["GET"])
def get_chunks():
    """List chunks for a document: /chunks?doc_id=123"""
    try:
        doc_id = int(request.args.get("doc_id", "0"))
    except ValueError:
        return jsonify({"error": "invalid doc_id"}), 400
    if not doc_id:
        return jsonify({"error": "doc_id required"}), 400

    chunks = get_chunks_for_doc(doc_id)
    out = [{
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "chunk_index": c.chunk_index,
        "token_count": c.token_count,
        "tok_ver": c.tok_ver,
        "seg_ver": c.seg_ver,
        "text": (lambda _t: _t[:300] + ("..." if len(_t) > 300 else ""))(str(getattr(c, "text", "") or "")),
    } for c in chunks]
    return jsonify({"doc_id": doc_id, "count": len(out), "chunks": out}), 200


@api_bp.route("/manifest", methods=["GET"])
def get_manifest():
    """Return manifest.jsonl lines for a document if present on local storage for compatibility."""
    doc_id = request.args.get("doc_id")
    if not doc_id:
        return jsonify({"error": "doc_id required"}), 400
    _, _, meta_dir = _upload_dirs()
    path = os.path.join(meta_dir, str(doc_id), "manifest.jsonl")
    if not os.path.exists(path):
        return jsonify({"doc_id": int(doc_id), "manifest": []}), 200
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(x) for x in f.read().splitlines() if x.strip()]
    return jsonify({"doc_id": int(doc_id), "manifest": lines}), 200


@api_bp.route("/embeddings/generate", methods=["POST"])
def generate_embeddings_endpoint():
    """Generate and store embeddings for all missing chunks of a document.

    Request JSON: { "doc_id": int, "model": "text-embedding-3-large" }
    Returns: { doc_id, model, generated }
    """
    data = request.get_json(force=True) or {}
    doc_id = data.get("doc_id")
    model = data.get("model") or os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    if not doc_id:
        return jsonify({"error": "doc_id required"}), 400
    try:
        generated = generate_and_store_embeddings(doc_id=str(doc_id), model=model)
        return jsonify({"doc_id": int(doc_id), "model": model, "generated": generated}), 200
    except Exception as e:
        logger.exception("Embedding generation failed: %s", e)
        return jsonify({"error": "failed to generate embeddings"}), 500


@api_bp.route("/query", methods=["POST"])
def query():
    """Vector search against chunk embeddings (Vertex AI backend).

    Request JSON:
      query: str (required)
      top_k: int (optional, default 5)
      filters: { doc_type?: str, jurisdiction?: str } (optional)
    """
    payload = request.get_json(force=True) or {}
    question = payload.get("query")
    top_k = int(payload.get("top_k", 5))
    filters = payload.get("filters") or {}
    f_doc_type = (filters.get("doc_type") or '').strip() if isinstance(filters, dict) else ''
    f_jurisdiction = (filters.get("jurisdiction") or '').strip() if isinstance(filters, dict) else ''
    if not question:
        return jsonify({"error": "query required"}), 400

    # Vertex AI Vector Search Integration
    project_id = os.getenv("VERTEX_PROJECT_ID")
    location = os.getenv("VERTEX_REGION")
    index_endpoint_id = os.getenv("VERTEX_INDEX_ENDPOINT_ID")
    deployed_index_id = os.getenv("VERTEX_DEPLOYED_INDEX_ID")

    if not all([project_id, location, index_endpoint_id, deployed_index_id]):
        logger.error("Vertex AI environment variables not configured.")
        return jsonify({"error": "Vector search is not configured."}), 501

    approx_multiplier = payload.get("approx_multiplier")
    approx_count = payload.get("approx_count")

    try:
        neighbors = find_nearest_neighbors_advanced(
            project_id=str(project_id),
            location=str(location),
            index_endpoint_id=str(index_endpoint_id),
            deployed_index_id=str(deployed_index_id),
            query=question,
            num_neighbors=top_k,
            approx_multiplier=float(approx_multiplier) if approx_multiplier is not None else None,
            approx_count=int(approx_count) if approx_count is not None else None,
        )
        if not neighbors:
            neighbors = find_nearest_neighbors(
                project_id=str(project_id),
                location=str(location),
                index_endpoint_id=str(index_endpoint_id),
                deployed_index_id=str(deployed_index_id),
                query=question,
                num_neighbors=top_k,
            )

        if not neighbors:
            return jsonify({"query": question, "results": [], "filters": filters}), 200

        neighbor_ids = [int(n_id) for n_id, _ in neighbors]
        chunks = get_chunks_by_ids(neighbor_ids)
        chunk_map = {}
        meta_map = {}
        for chunk in chunks:
            cid = int(getattr(chunk, "chunk_id"))
            chunk_map[cid] = getattr(chunk, "text", "")
            doc_obj = getattr(chunk, "document", None)
            if doc_obj is not None:
                meta_map[cid] = {
                    "doc_id": getattr(doc_obj, "doc_id", None),
                    "doc_type": getattr(doc_obj, "doc_type", None),
                    "jurisdiction": getattr(doc_obj, "jurisdiction", None),
                    "title": getattr(doc_obj, "title", None),
                }

        results = []
        for n_id, dist in neighbors:
            try:
                cid = int(n_id)
            except (TypeError, ValueError):
                continue
            meta = meta_map.get(cid, {})
            if f_doc_type and (meta.get("doc_type") or '').lower() != f_doc_type.lower():
                continue
            if f_jurisdiction and (meta.get("jurisdiction") or '').lower() != f_jurisdiction.lower():
                continue
            results.append({
                "chunk_id": cid,
                "text": chunk_map.get(cid, ""),
                "distance": dist,
                **meta
            })

        return jsonify({"query": question, "results": results, "filters": filters}), 200
    except Exception as e:
        logger.exception("Vector search query failed: %s", e)
        return jsonify({"error": "Failed to execute vector search."}), 500


# Health
@api_bp.route("/healthz", methods=["GET"])
def healthz():
    try:
        init_db()  # idempotent
        return jsonify({"ok": True}), 200
    except Exception as e:
        logger.exception("health check failed: %s", e)
        return jsonify({"ok": False}), 500
