# api.py (Cloud Run + PostgreSQL refactor)
import os
import json
from google.cloud import storage
from google.api_core.exceptions import NotFound
import filetype
from typing import Any, Dict
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from infrastructure.logger import get_logger

from infrastructure.database import (
    init_db,
    add_document,
    get_document_by_sha,
    get_chunks_for_doc,
    get_documents,
    get_document_chunk_counts,
    get_all_chunks,
    Document,
)
from infrastructure.embeddings import generate_and_store_embeddings, find_nearest_neighbors_advanced
from infrastructure.document_processor import (
    extract_text,
    extract_title_type_jurisdiction,
    upsert_manifest_record,
    sha256_text,
)
from infrastructure.chunker import chunk_doc
from infrastructure.vector_search import find_nearest_neighbors
from infrastructure.database import get_chunks_by_ids

api_bp = Blueprint("api", __name__, url_prefix="/api")
logger = get_logger()

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
GCS_DEFAULT_STORAGE_CLASS = "STANDARD"
GCS_DEFAULT_LOCATION = "US"
PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "tech-trans-rag")
GCS_URI = os.getenv("GCS_URI", "gs://tech-trans-rag-bucket")
GCS_BUCKET = os.getenv("GCS_BUCKET", "tech-trans-rag-bucket")
MANIFEST_DOC = os.getenv("MANIFEST_DOC", "manifest/manifest.jsonl")
CHUNKS_DOC = os.getenv("CHUNKS_DOC", "chunks/chunks.jsonl")

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_dir(bucket_name: str) -> Dict[str, Any]:
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(bucket_name)
        if not bucket.exists():
            bucket.storage_class = GCS_DEFAULT_STORAGE_CLASS
            new_bucket = client.create_bucket(bucket, location=GCS_DEFAULT_LOCATION)
            logger.info("Created new bucket %s in %s with class %s", new_bucket.name, new_bucket.location, new_bucket.storage_class)
            return {
                "status": "created", 
                "bucket": new_bucket.name,
                "location": new_bucket.location
            }
        else:
            logger.info("Bucket %s already exists", bucket_name)
            return {
                "status": "exists", 
                "bucket": bucket_name,
                "location": bucket.location,
            }
    except Exception as e:
        logger.error("Failed to create directory %s: %s", bucket_name, e)
        return {
            "status": "error",
            "error": str(e)
        }


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
    
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)

    clean_text = extract_text(file=file)
    content_sha = sha256_text(clean_text)


    doc = get_document_by_sha(content_sha)
    title = None
    raw_blob_path = os.getenv("UPLOAD_FOLDER", "corpus_raw/") + file.filename
    logger.info("Raw blob path: %s", raw_blob_path)
    txt_filename = file.filename.rsplit('.', 1)[0] + '.txt'
    clean_blob_path = os.getenv("CLEAN_FOLDER", "corpus_clean/") + txt_filename
    logger.info("Clean blob path: %s", clean_blob_path)
    doc_id_val = None

    if not doc:
        filename = secure_filename(file.filename)
        logger.info("Secured filename: %s", filename)
        doc_info = extract_title_type_jurisdiction(clean_text)
        title = doc_info.get("title") or file.filename.split('.')[0]
        if not doc_type and "doc_type" in doc_info:
            doc_type = doc_info.get("doc_type")
        if not jurisdiction and "jurisdiction" in doc_info:
            jurisdiction = doc_info.get("jurisdiction")
        raw_blob = bucket.blob(raw_blob_path)
        file.stream.seek(0)
        raw_blob.upload_from_file(file_obj=file, content_type=file.mimetype)
        logger.info("Uploaded raw file to GCS bucket %s as %s", bucket.name, raw_blob_path)

        clean_blob = bucket.blob(clean_blob_path)
        clean_blob.upload_from_string(clean_text, content_type="text/plain")
        logger.info("Uploaded clean text to GCS bucket %s as %s", bucket.name, clean_blob_path)
        doc = add_document(sha256=content_sha, title=title, source_path=raw_blob.public_url, doc_type=doc_type, jurisdiction=jurisdiction)

        file.stream.seek(0)
        size = file.stream.tell()
        file.stream.seek(0)

        header = file.stream.read(2048)
        file.stream.seek(0)
        kind = filetype.guess(header)
        content_type = kind.mime if kind else 'application/octet-stream'

        doc_id_val = getattr(doc, "doc_id", None)

        manifest_record = upsert_manifest_record(text=clean_text, size=str(size), doc_id=str(doc_id_val) or "", source_path=raw_blob.public_url, clean_path=clean_blob.public_url, sha256=content_sha, title=title, jurisdiction=jurisdiction or "", doc_type=doc_type or "", content_type=content_type)

        logger.info("Upserted manifest record: %s", manifest_record)
        governing_law = manifest_record.get("governing_law")
        party_roles = manifest_record.get("party_roles")
        industry = manifest_record.get("industry")
        effective_date = manifest_record.get("effective_date")
        doc = add_document(sha256=content_sha, governing_law=governing_law, party_roles=party_roles, industry=industry, effective_date=effective_date)
    else:
        # Enrich existing if metadata blank (add_document handles enrichment when doc exists)
        try:
            doc = add_document(sha256=content_sha, doc_type=doc_type, jurisdiction=jurisdiction)
        except Exception:
            pass

    # Chunk now. Persist chunks into Postgres via add_chunk.
    try:
        chunks = chunk_doc(text=clean_text, doc_id=str(doc_id_val))
    except Exception as e:
        logger.exception("Chunking failed: %s", e)
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
        "text": str(getattr(c, "text", "") or ""),
    } for c in chunks]
    return jsonify({"doc_id": doc_id, "count": len(out), "chunks": out}), 200

@api_bp.route("/documents", methods=["GET"])
def list_documents():
    """List documents with pagination and basic counts.

    Query params:
      limit (default 100, max 500)
      offset (default 0)
    """
    try:
        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))
    except ValueError:
        return jsonify({"error": "invalid pagination"}), 400
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    docs = get_documents(limit=limit, offset=offset)
    ids = [int(getattr(d, 'doc_id')) for d in docs]
    counts = get_document_chunk_counts(ids)
    out = []
    for d in docs:
        out.append({
            "doc_id": int(getattr(d, 'doc_id')),
            "title": d.title,
            "sha256": d.sha256,
            "doc_type": getattr(d, "doc_type", None),
            "jurisdiction": getattr(d, "jurisdiction", None),
            "created_at": d.created_at.isoformat() if getattr(d, 'created_at', None) else None,
            "chunk_count": counts.get(int(getattr(d, 'doc_id')), 0),
        })
    return jsonify({"documents": out, "count": len(out), "limit": limit, "offset": offset}), 200


@api_bp.route("/chunks/all", methods=["GET"])
def list_all_chunks():
    """List global chunks slice (paginated). Query params: limit (<=500), offset."""
    try:
        limit = int(request.args.get("limit", 200))
        offset = int(request.args.get("offset", 0))
    except ValueError:
        return jsonify({"error": "invalid pagination"}), 400
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    rows = get_all_chunks(limit=limit, offset=offset)
    out = []
    for c in rows:
        doc = getattr(c, 'document', None)
        out.append({
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "chunk_index": c.chunk_index,
            "token_count": c.token_count,
            "tok_ver": c.tok_ver,
            "seg_ver": c.seg_ver,
            "text": str(getattr(c, 'text', '') or ''),
            "doc_type": getattr(doc, 'doc_type', None) if doc else None,
            "jurisdiction": getattr(doc, 'jurisdiction', None) if doc else None,
            "title": getattr(doc, 'title', None) if doc else None,
        })
    return jsonify({"chunks": out, "count": len(out), "limit": limit, "offset": offset}), 200


@api_bp.route("/manifest", methods=["GET"])
def get_manifest():
    """Return manifest.jsonl lines for a document if present on local storage for compatibility."""

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET) 
    blob = bucket.blob(MANIFEST_DOC)
    try:
        content = blob.download_as_text(encoding="utf-8")
    except NotFound:
        return []

    return jsonify(json.loads(content)), 200


@api_bp.route("/manifest", methods=["PATCH"])
def patch_manifest():
    """Patch/update one manifest record by doc_id.

    Request JSON: { "doc_id": <id>, "updates": { field: value, ... } }
    Disallowed fields: doc_id, source_path (cannot be modified here).
    Writes the full manifest JSON array back to GCS.
    Returns: { "updated": record } or error.
    """
    payload = request.get_json(force=True) or {}
    doc_id = payload.get("doc_id")
    updates = payload.get("updates") or {}
    if doc_id is None:
        return jsonify({"error": "doc_id required"}), 400
    if not isinstance(updates, dict):
        return jsonify({"error": "updates must be object"}), 400
    # Prevent editing disallowed keys
    for k in ["doc_id", "source_path"]:
        if k in updates:
            updates.pop(k, None)
    if not updates:
        return jsonify({"error": "no editable fields supplied"}), 400

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(MANIFEST_DOC)
    try:
        raw = blob.download_as_text(encoding="utf-8")
    except NotFound:
        return jsonify({"error": "manifest not found"}), 404
    try:
        data = json.loads(raw)
    except Exception:
        return jsonify({"error": "manifest unreadable"}), 500
    if not isinstance(data, list):
        return jsonify({"error": "manifest format invalid"}), 500

    str_doc_id = str(doc_id)
    updated_record = None
    changed = False
    for rec in data:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("doc_id")) == str_doc_id:
            # Apply updates
            for k, v in updates.items():
                old_val = rec.get(k)
                if old_val != v:
                    rec[k] = v
                    changed = True
            updated_record = rec
            break

    if not updated_record:
        return jsonify({"error": "record not found"}), 404
    if not changed:
        return jsonify({"updated": updated_record, "changed": False}), 200

    try:
        blob.upload_from_string(json.dumps(data, ensure_ascii=False, indent=2), content_type="application/json")
        # Also update the database record
        if updated_record and "sha256" in updated_record:
            sha256 = updated_record["sha256"]
            db_updates = {k: v for k, v in updates.items() if hasattr(Document, k)}
            if db_updates:
                add_document(sha256=sha256, **db_updates)
                logger.info("Enriched document %s in database with new metadata.", doc_id)

    except Exception as e:
        logger.exception("Failed to write manifest or update database: %s", e)
        return jsonify({"error": "failed to persist changes"}), 500

    return jsonify({"updated": updated_record, "changed": True}), 200


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
