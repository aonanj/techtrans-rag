"""Embedding utilities.

Generates embeddings for chunk records lacking a stored vector and writes
results to the SQLite `embeddings` table. Also upserts into a Vertex AI 
Matching Engine index.

Primary entrypoint:
    generate_and_store_embeddings(doc_id: str | None = None,
                                   model: str | None = None,
                                   batch_size: int = 64) -> int

Requirements / Assumptions:
* Uses OpenAI Embeddings API (model default: text-embedding-3-large).
* Stores vectors as little-endian float32 binary blobs (dimension recorded).
* Uses Vertex AI Matching Engine for vector indexing. Requires environment
  variables for project, region, and index endpoint.
* Current DB schema has `chunk_id` as PRIMARY KEY in embeddings table, so only
  one embedding per chunk is effectively stored despite a `model` column.

Graceful degradation:
* If openai or google-cloud-aiplatform packages are missing or API calls fail, 
  logs an error and aborts/skips steps.
* Batches requests; naive exponential backoff on rate / transient errors.

Future improvements:
* Streaming ingestion & async concurrency.
* Token counting for dynamic batch sizing.
"""
from __future__ import annotations

import array
import os
import time
from typing import Iterable, List, Optional, Tuple, Any
from openai import OpenAI
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndexEndpoint,
    MatchingEngineIndex,
)
try:  # Optional advanced GAPIC client for approximate neighbor tuning
    from google.cloud import aiplatform_v1  # type: ignore
except Exception:  # pragma: no cover
    aiplatform_v1 = None  # type: ignore

from . import database
from .logger import get_logger

logger = get_logger()

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

def serialize_vector(vec: List[float]) -> bytes:
    """Pack list of floats into a binary blob (float32 little-endian)."""
    arr = array.array("f", vec)
    return arr.tobytes()


def deserialize_vector(blob: bytes) -> List[float]:
    """Unpack binary blob into a list of floats."""
    arr = array.array("f")
    arr.frombytes(blob)
    return list(arr)


def _get_vertex_index_endpoint() -> Optional[MatchingEngineIndexEndpoint]:
    """Return a Vertex AI Matching Engine Index Endpoint instance."""
    if not all([aiplatform, MatchingEngineIndexEndpoint]):
        logger.warning(
            "google-cloud-aiplatform not installed; skipping vector index operations"
        )
        return None

    project_id = os.getenv("VERTEX_PROJECT_ID")
    region = os.getenv("VERTEX_REGION")
    index_endpoint_id = os.getenv("VERTEX_INDEX_ENDPOINT_ID")

    # Explicit checks (avoid all([...]) so type checkers narrow to str)
    if not (project_id and region and index_endpoint_id):
        logger.warning(
            "Vertex AI environment variables not set; skipping vector index operations"
        )
        return None

    try:
        aiplatform.init(project=project_id, location=region)
        return MatchingEngineIndexEndpoint(index_endpoint_name=index_endpoint_id)
    except Exception as e:
        logger.error("Failed to initialize Vertex AI Index Endpoint: %s", e)
        return None


def fetch_chunks_missing_embeddings(
    db: Any, model: str, doc_id: Optional[str] = None
) -> List[Tuple[str, str]]:
    """Return list of (chunk_id, text) needing embeddings for given model."""
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


def upsert_embedding(
    db: Any, chunk_id: str, model: str, vector: List[float]
):
    """Insert or replace an embedding in the local SQLite database."""
    blob = serialize_vector(vector)
    dim = len(vector)
    db.execute(
        """REPLACE INTO embeddings(chunk_id, model, dim, vector) VALUES (?,?,?,?)""",
        (chunk_id, model, dim, blob),
    )


def batched(
    iterable: List[Tuple[str, str]], n: int
) -> Iterable[List[Tuple[str, str]]]:
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def _ensure_openai_client() -> Optional[OpenAI]:
    """Ensure OpenAI client is available and configured."""
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


def generate_embeddings(
    texts: List[str],
    model: str | None = None,
    max_retries: int = 5,
) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    model = model or DEFAULT_MODEL
    client = _ensure_openai_client()
    if not client:
        return []

    attempt = 0
    while True:
        try:
            resp = client.embeddings.create(model=model, input=texts)
            vectors = [d.embedding for d in resp.data]
            if len(vectors) != len(texts):
                raise RuntimeError("Mismatch between returned embeddings and input size")
            return vectors
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "Failed embedding batch after %d attempts: %s", attempt - 1, e
                )
                return []
            sleep_for = min(2**attempt, 30)
            logger.warning(
                "Embedding batch error (%s). Retry %d/%d in %.1fs",
                e,
                attempt,
                max_retries,
                sleep_for,
            )
            time.sleep(sleep_for)
    return []


def generate_embedding(
    text: str,
    model: str | None = None,
    max_retries: int = 5,
) -> List[float]:
    """Generate an embedding for a single text."""
    embeddings = generate_embeddings([text], model=model, max_retries=max_retries)
    return embeddings[0] if embeddings else []


def generate_and_store_embeddings(
    doc_id: str | None = None,
    model: str | None = None,
    batch_size: int = 64,
    max_retries: int = 5,
) -> int:
    """Generate embeddings for missing chunks and store them (and optionally index)."""
    model = model or DEFAULT_MODEL
    db = database.get_db()
    missing = fetch_chunks_missing_embeddings(db, model=model, doc_id=doc_id)
    if not missing:
        logger.info("No missing embeddings (doc_id=%s, model=%s)", doc_id, model)
        return 0

    client = _ensure_openai_client()
    if client is None:
        return 0

    vertex_index_endpoint = _get_vertex_index_endpoint()
    index_id = os.getenv("VERTEX_INDEX_ID") if vertex_index_endpoint else None
    vertex_index = None
    if index_id:
        try:
            vertex_index = MatchingEngineIndex(index_name=index_id)
        except Exception as e:
            logger.error("Failed to initialize Vertex AI Index: %s", e)

    total_written = 0
    for batch in batched(missing, batch_size):
        ids, texts = zip(*batch)
        vectors = generate_embeddings(list(texts), model=model, max_retries=max_retries)
        if not vectors:
            logger.error("Failed to generate embeddings for batch, skipping.")
            continue

        # Store embeddings locally
        for cid, vec in zip(ids, vectors):
            upsert_embedding(db, cid, model, vec)

        # Upsert to Vertex AI (prefer direct index object if available)
        if vertex_index:
            try:
                datapoints = [
                    {"datapoint_id": cid, "feature_vector": vec}
                    for cid, vec in zip(ids, vectors)
                ]
                vertex_index.upsert_datapoints(datapoints=datapoints)
            except Exception as e:
                logger.error("Vertex AI upsert failed: %s", e)
        elif vertex_index_endpoint and index_id:
            # MatchingEngineIndexEndpoint has no upsert_datapoints method; requires MatchingEngineIndex.
            logger.warning(
                "Vertex index object unavailable; skipping upsert for %d datapoints",
                len(ids),
            )

        db.commit()
        total_written += len(ids)
        logger.info(
            "Embedded %d chunks (running total %d)", len(ids), total_written
        )

    return total_written


__all__ = [
    "generate_embeddings",
    "generate_embedding",
    "generate_and_store_embeddings",
    "fetch_chunks_missing_embeddings",
    "find_nearest_neighbors_advanced",
]


def _build_full_index_endpoint_name(project_id: str, location: str, index_endpoint_id: str) -> str:
    if index_endpoint_id.startswith("projects/"):
        return index_endpoint_id
    return f"projects/{project_id}/locations/{location}/indexEndpoints/{index_endpoint_id}"


def find_nearest_neighbors_advanced(
    project_id: str,
    location: str,
    index_endpoint_id: str,
    deployed_index_id: str,
    query: str,
    num_neighbors: int = 5,
    approx_multiplier: float | None = None,
    approx_count: int | None = None,
) -> list[tuple[str, float]]:
    """Find nearest neighbors with optional control over approximate neighbor count.

    Falls back to basic MatchingEngineIndexEndpoint.find_neighbors if GAPIC advanced
    client is unavailable or an error occurs.
    """
    if num_neighbors <= 0:
        return []
    aiplatform.init(project=project_id, location=location)
    query_embedding = generate_embedding(query)
    if not query_embedding:
        logger.error("Failed to generate embedding for query; returning empty result")
        return []

    # Attempt advanced path (approximate_neighbor_count) if client available
    if aiplatform_v1 and hasattr(aiplatform_v1, "MatchServiceClient"):
        try:
            full_ep = _build_full_index_endpoint_name(project_id, location, index_endpoint_id)
            if approx_count and approx_count < num_neighbors:
                logger.warning(
                    "approx_count (%d) < num_neighbors (%d); coercing", approx_count, num_neighbors
                )
            if approx_count:
                approximate_neighbor_count = max(approx_count, num_neighbors)
            elif approx_multiplier:
                approximate_neighbor_count = max(int(num_neighbors * approx_multiplier), num_neighbors)
            else:
                approximate_neighbor_count = min(max(num_neighbors * 4, num_neighbors), 200)

            me_client = aiplatform_v1.MatchServiceClient()
            query_obj = aiplatform_v1.FindNeighborsRequest.Query(
                neighbor_count=num_neighbors,
                approximate_neighbor_count=approximate_neighbor_count,
                datapoint=aiplatform_v1.IndexDatapoint(feature_vector=query_embedding),
            )
            request = aiplatform_v1.FindNeighborsRequest(
                index_endpoint=full_ep,
                deployed_index_id=deployed_index_id,
                queries=[query_obj],
            )
            response = me_client.find_neighbors(request=request)
            neighbors: list[tuple[str, float]] = []
            if response.nearest_neighbors and response.nearest_neighbors[0].neighbors:
                for n in response.nearest_neighbors[0].neighbors:
                    neighbors.append((n.datapoint.datapoint_id, n.distance))
            return neighbors
        except Exception as e:  # pragma: no cover
            logger.warning("Advanced neighbor search failed (%s); falling back to basic API", e)

    # Basic fallback
    try:
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=index_endpoint_id
        )
        resp = index_endpoint.find_neighbors(
            deployed_index_id=deployed_index_id,
            queries=[query_embedding],
            num_neighbors=num_neighbors,
        )
        neighbors_basic: list[tuple[str, float]] = []
        if resp and resp[0]:
            for n in resp[0]:
                dist = 0.0 if n.distance is None else float(n.distance)
                neighbors_basic.append((n.id, dist))
        return neighbors_basic
    except Exception as e:  # pragma: no cover
        logger.error("Basic neighbor search failed: %s", e)
        return []
