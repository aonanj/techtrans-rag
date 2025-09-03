"""Embedding utilities.

Generates embeddings for chunk records lacking a stored vector and writes
results to the SQLite `embeddings` table (serving as a lightweight vector DB).

Primary entrypoint:
    generate_and_store_embeddings(doc_id: str | None = None,
                                   model: str | None = None,
                                   batch_size: int = 64) -> int

Requirements / Assumptions:
* Uses OpenAI Embeddings API (model default: text-embedding-3-large).
* Stores vectors as little-endian float32 binary blobs (dimension recorded).
* Current DB schema has `chunk_id` as PRIMARY KEY in embeddings table, so only
  one embedding per chunk is effectively stored despite a `model` column.
  (If multi-model support is desired, the schema must be altered to include
  a composite primary key or separate table.)

Graceful degradation:
* If openai package is missing or API call fails, logs an error and aborts.
* Batches requests; naive exponential backoff on rate / transient errors.

Future improvements:
* Optional FAISS / Chroma / LanceDB index build alongside SQL storage.
* Streaming ingestion & async concurrency.
* Token counting for dynamic batch sizing.
"""
from __future__ import annotations

from typing import List, Iterable, Tuple, Optional, Dict
import os
import time
import sqlite3
import array
import chromadb
from .logger import get_logger
from . import database

logger = get_logger()

try:  # Soft dependency to avoid breaking unrelated workflows
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")


def serialize_vector(vec: List[float]) -> bytes:
    """Pack list of floats into a binary blob (float32 little-endian)."""
    arr = array.array('f', vec)
    return arr.tobytes()


def deserialize_vector(blob: bytes) -> List[float]:
    arr = array.array('f')
    arr.frombytes(blob)
    return list(arr)


def _get_chroma_collection(collection_name: str = "chunks", persist_dir: Optional[str] = None) -> Optional[object]:
    """Return (or create) a Chroma collection backed by persistent directory.

    Returns None if chromadb is not installed.
    """
    if chromadb is None:
        logger.warning("Chroma not installed; skipping vector index operations")
        return None
    persist_dir = persist_dir or os.getenv("INDEX_FOLDER", "./data/index")
    os.makedirs(persist_dir, exist_ok=True)
    try:
        client = chromadb.PersistentClient(path=persist_dir)  # type: ignore
        return client.get_or_create_collection(name=collection_name)
    except Exception as e:  # pragma: no cover
        logger.error("Failed to obtain Chroma collection '%s': %s", collection_name, e)
        return None


def _fetch_chunk_metadata(db: sqlite3.Connection, chunk_ids: List[str]) -> Dict[str, Dict[str, Optional[str]]]:
    if not chunk_ids:
        return {}
    placeholders = ",".join(["?"] * len(chunk_ids))
    rows = db.execute(
        f"SELECT chunk_id, doc_id, page_start, page_end, section FROM chunks WHERE chunk_id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for r in rows:
        out[r[0]] = {
            "doc_id": r[1],
            "page_start": str(r[2]) if r[2] is not None else None,
            "page_end": str(r[3]) if r[3] is not None else None,
            "section": r[4],
        }
    return out


def fetch_chunks_missing_embeddings(db: sqlite3.Connection, model: str, doc_id: Optional[str] = None) -> List[Tuple[str, str]]:
    """Return list of (chunk_id, text) needing embeddings for given model.

    NOTE: Schema defines chunk_id primary key in embeddings table; we treat
    absence of any row for chunk_id as missing, ignoring model uniqueness.
    If a row exists with a different model, it will be overwritten later.
    """
    if doc_id:
        rows = db.execute(
            """
            SELECT c.chunk_id, c.text
            FROM chunks c
            LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id
            WHERE c.doc_id = ? AND e.chunk_id IS NULL
            ORDER BY c.chunk_id
            """,
            (doc_id,),
        ).fetchall()
    else:
        rows = db.execute(
            """
            SELECT c.chunk_id, c.text
            FROM chunks c
            LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id
            WHERE e.chunk_id IS NULL
            ORDER BY c.chunk_id
            """,
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


def upsert_embedding(db: sqlite3.Connection, chunk_id: str, model: str, vector: List[float]):
    blob = serialize_vector(vector)
    dim = len(vector)
    # Because chunk_id is PRIMARY KEY, REPLACE semantics allow only one embedding per chunk.
    db.execute(
        """REPLACE INTO embeddings(chunk_id, model, dim, vector) VALUES (?,?,?,?)""",
        (chunk_id, model, dim, blob),
    )


def batched(iterable: List[Tuple[str, str]], n: int) -> Iterable[List[Tuple[str, str]]]:
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def _ensure_client() -> Optional[object]:
    if OpenAI is None:
        logger.error("openai package not installed; cannot generate embeddings")
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set; cannot generate embeddings")
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:  # pragma: no cover
        logger.error("Failed to init OpenAI client: %s", e)
        return None


def generate_and_store_embeddings(doc_id: str | None = None, model: str | None = None, batch_size: int = 64, max_retries: int = 5) -> int:
    """Generate embeddings for missing chunks and store them.

    Args:
        doc_id: If provided, limit to a single document's chunks.
        model: Embedding model name (defaults to configured / env DEFAULT_MODEL).
        batch_size: Number of texts per API call.
        max_retries: Retry attempts for transient API failures.

    Returns:
        Number of embeddings successfully written.
    """
    model = model or DEFAULT_MODEL
    db = database.get_db()
    missing = fetch_chunks_missing_embeddings(db, model=model, doc_id=doc_id)
    if not missing:
        logger.info("No missing embeddings (doc_id=%s, model=%s)", doc_id, model)
        return 0

    client = _ensure_client()
    if client is None:
        return 0

    total_written = 0
    for batch in batched(missing, batch_size):
        ids, texts = zip(*batch)
        attempt = 0
        while True:
            try:
                resp = client.embeddings.create(model=model, input=list(texts))  # type: ignore[attr-defined]
                vectors = [d.embedding for d in resp.data]  # type: ignore
                if len(vectors) != len(ids):
                    raise RuntimeError("Mismatch between returned embeddings and input size")
                # Upsert in SQL + Chroma in same loop to reuse vectors
                chroma_collection = _get_chroma_collection()
                meta_map = _fetch_chunk_metadata(db, list(ids)) if chroma_collection else {}
                docs: List[str] = []
                metas: List[Dict[str, Optional[str]]] = []
                for cid, vec in zip(ids, vectors):
                    upsert_embedding(db, cid, model, vec)
                    if chroma_collection:
                        md = meta_map.get(cid, {})
                        metas.append({
                            "doc_id": md.get("doc_id"),
                            "page_start": md.get("page_start"),
                            "page_end": md.get("page_end"),
                            "section": md.get("section"),
                            "model": model,
                        })
                        # Fetch text for documents argument from already available texts tuple
                        # We rely on order alignment with ids/texts
                        # (ids, texts) zipped above, so reuse texts
                        # Build docs list later after loop over zipped vectors
                if chroma_collection:
                    # Build docs using original texts mapping by id (ids aligned with texts)
                    id_to_text = dict(zip(ids, texts))
                    docs = [id_to_text[cid] for cid in ids]
                    try:
                        chroma_collection.add(  # type: ignore[attr-defined]
                            ids=list(ids),
                            embeddings=vectors,
                            metadatas=metas,
                            documents=docs,
                        )  # type: ignore
                    except Exception as ce:  # pragma: no cover
                        logger.warning("Chroma add failed (%s); attempting update", ce)
                        try:
                            chroma_collection.update(  # type: ignore[attr-defined]
                                ids=list(ids),
                                embeddings=vectors,
                                metadatas=metas,
                                documents=docs,
                            )  # type: ignore
                        except Exception as ce2:
                            logger.error("Chroma update failed: %s", ce2)
                db.commit()
                total_written += len(ids)
                logger.info("Embedded %d chunks (running total %d)", len(ids), total_written)
                break
            except Exception as e:  # pragma: no cover (broad but logged)
                attempt += 1
                if attempt > max_retries:
                    logger.error("Failed embedding batch after %d attempts: %s", attempt - 1, e)
                    break
                sleep_for = min(2 ** attempt, 30)
                logger.warning("Embedding batch error (%s). Retry %d/%d in %.1fs", e, attempt, max_retries, sleep_for)
                time.sleep(sleep_for)

    return total_written


__all__ = [
    "generate_and_store_embeddings",
    "fetch_chunks_missing_embeddings",
]


def build_chroma_index(doc_id: str | None = None, model: str | None = None, batch_size: int = 200) -> int:
    """Create (or rebuild missing parts of) a Chroma index from current SQL embeddings.

    Args:
        doc_id: Limit to a single document if provided.
        model: Filter embeddings by model (default any stored model).
        batch_size: Number of vectors per ingestion batch.

    Returns:
        Number of vectors added/updated in Chroma.
    """
    collection = _get_chroma_collection()
    if collection is None:
        return 0
    db = database.get_db()
    clauses = []
    params: List[str] = []
    if doc_id:
        clauses.append("c.doc_id = ?")
        params.append(doc_id)
    if model:
        clauses.append("e.model = ?")
        params.append(model)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = db.execute(
        f"""
        SELECT e.chunk_id, e.model, e.dim, e.vector, c.text, c.doc_id, c.page_start, c.page_end, c.section
        FROM embeddings e
        JOIN chunks c ON c.chunk_id = e.chunk_id
        {where}
        ORDER BY e.chunk_id
        """,
        params,
    ).fetchall()
    if not rows:
        logger.info("No embeddings found to index (doc_id=%s model=%s)", doc_id, model)
        return 0
    count = 0
    batch_ids: List[str] = []
    batch_vecs: List[List[float]] = []
    batch_docs: List[str] = []
    batch_meta: List[Dict[str, Optional[str]]] = []
    for r in rows:
        cid, mname, dim, blob, text, did, ps, pe, section = r
        vec = deserialize_vector(blob)
        if len(vec) != dim:
            logger.warning("Vector length mismatch for %s (expected %s got %s)", cid, dim, len(vec))
        batch_ids.append(cid)
        batch_vecs.append(vec)
        batch_docs.append(text)
        batch_meta.append({
            "doc_id": did,
            "model": mname,
            "page_start": str(ps) if ps is not None else None,
            "page_end": str(pe) if pe is not None else None,
            "section": section,
        })
        if len(batch_ids) >= batch_size:
            try:
                collection.add(ids=batch_ids, embeddings=batch_vecs, metadatas=batch_meta, documents=batch_docs)  # type: ignore[attr-defined]
            except Exception:
                try:
                    collection.update(ids=batch_ids, embeddings=batch_vecs, metadatas=batch_meta, documents=batch_docs)  # type: ignore[attr-defined]
                except Exception as e:
                    logger.error("Failed adding/updating batch to Chroma: %s", e)
            count += len(batch_ids)
            batch_ids, batch_vecs, batch_docs, batch_meta = [], [], [], []
    if batch_ids:
        try:
            collection.add(ids=batch_ids, embeddings=batch_vecs, metadatas=batch_meta, documents=batch_docs)  # type: ignore[attr-defined]
        except Exception:
            try:
                collection.update(ids=batch_ids, embeddings=batch_vecs, metadatas=batch_meta, documents=batch_docs)  # type: ignore[attr-defined]
            except Exception as e:
                logger.error("Failed final batch add/update to Chroma: %s", e)
        count += len(batch_ids)
    logger.info("Indexed %d vectors into Chroma (doc_id=%s model=%s)", count, doc_id, model)
    return count

__all__.append("build_chroma_index")
