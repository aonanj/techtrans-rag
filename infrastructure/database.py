"""
Cloud SQL (PostgreSQL) database layer for Cloud Run.

Features:
- SQLAlchemy engine with small pools for serverless
- Works with Private IP (host/port) OR Cloud SQL Python Connector (IAM or password)
- Minimal ORM schema for documents, chunks, and embeddings
- Drop-in helpers mirroring prior sqlite-style operations

Env vars (set in Cloud Run):
  DB_NAME=appdb
  DB_USER=appuser                     # not needed if IAM DB auth is used
  DB_PASSWORD=...                     # inject via Secret Manager if using password auth
  DB_HOST=10.x.x.x                    # Private IP address of the Cloud SQL instance
  DB_PORT=5432                        # optional
  DB_USE_CONNECTOR=true|false         # true to use Cloud SQL Python Connector
  CLOUD_SQL_CONNECTION_NAME=proj:region:instance  # required if DB_USE_CONNECTOR=true
  DB_IAM_AUTH=true|false              # true to use IAM DB auth via connector (no password)

Notes:
- Keep connection counts low on Cloud Run: pool_size=1, max_overflow=2.
- If you enabled IAM DB auth, create a DB user mapped to your Run service account and GRANT privileges.
"""
import os
import contextlib
import threading
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, ForeignKey, LargeBinary, Index, text, func
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, scoped_session, joinedload

try:
    from infrastructure.logger import get_logger  # reuse central logger if available
    _logger = get_logger()
except Exception:  # pragma: no cover - fallback minimal logger
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    _logger = _logging.getLogger("db")
from flask import g
from google.cloud.sql.connector import Connector, IPTypes

# ---------------------------------------------------------------------------
# Versioning for corpus mechanics
# ---------------------------------------------------------------------------
TOK_VER = int(os.getenv("TOK_VER", "1"))
SEG_VER = int(os.getenv("SEG_VER", "1"))

# ---------------------------------------------------------------------------
# Engine factory
_ENGINE = None
_SESSION_FACTORY = None
_LOCK = threading.Lock()
Base = declarative_base()

def _build_engine():
    use_connector = os.getenv("DB_USE_CONNECTOR", "false").lower() == "true"
    use_connector = os.getenv("DB_USE_CONNECTOR", "false").lower() == "true"
    db_name = os.getenv("DB_NAME", "appdb")
    pool_kwargs = dict(pool_size=1, max_overflow=2, pool_pre_ping=True, pool_recycle=1800)

    if use_connector:
        if Connector is None:
            raise RuntimeError("google-cloud-sql-connector not installed but DB_USE_CONNECTOR=true")
        conn_name = os.environ["CLOUD_SQL_CONNECTION_NAME"]  # proj:region:instance
        iam_auth = os.getenv("DB_IAM_AUTH", "false").lower() == "true"
        db_user = os.getenv("DB_USER")  # if iam_auth is true, this should be the IAM principal email
        db_password = os.getenv("DB_PASSWORD")  # optional when iam_auth=true

        connector = Connector(ip_type=IPTypes.PRIVATE)

        def getconn():
            return connector.connect(
                conn_name,
                "pg8000",
                user=db_user,
                password=None if iam_auth else db_password,
                db=db_name,
                enable_iam_auth=iam_auth,
            )

        # Creator-based engine; SQLAlchemy URL is a placeholder
        return create_engine("postgresql+pg8000://", creator=getconn, **pool_kwargs)

    # Private IP TCP path
    db_user = os.getenv("DB_USER", "appuser")
    db_password = os.getenv("DB_PASSWORD", "")
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("DB_PORT", "5432"))

    # URL-encode password if needed
    from urllib.parse import quote_plus
    pwd = quote_plus(db_password) if db_password else ""
    auth = f"{db_user}:{pwd}@" if db_password or db_user else ""
    url = f"postgresql+pg8000://{auth}{host}:{port}/{db_name}"
    return create_engine(url, **pool_kwargs)

def _ensure_engine():
    global _ENGINE, _SESSION_FACTORY
    if _ENGINE is None:
        with _LOCK:
            if _ENGINE is None:
                _ENGINE = _build_engine()
                # expire_on_commit=False so returned ORM instances retain attribute values
                _SESSION_FACTORY = scoped_session(sessionmaker(
                    bind=_ENGINE,
                    autoflush=False,
                    autocommit=False,
                    expire_on_commit=False,
                ))
    return _ENGINE

# ---------------------------------------------------------------------------
# ORM schema
# ---------------------------------------------------------------------------
class Document(Base):
    __tablename__ = "documents"
    doc_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(512), nullable=True)
    source_path = Column(Text, nullable=True)
    sha256 = Column(String(64), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    # Optional metadata filters
    doc_type = Column(String(128), nullable=True)
    jurisdiction = Column(String(64), nullable=True)

    chunks = relationship("Chunk", cascade="all, delete-orphan", back_populates="document")

class Chunk(Base):
    __tablename__ = "chunks"
    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(Integer, ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)  # position within the doc
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    section = Column(Text, nullable=True)
    text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=True)
    tok_ver = Column(Integer, default=TOK_VER, nullable=False)
    seg_ver = Column(Integer, default=SEG_VER, nullable=False)

    document = relationship("Document", back_populates="chunks")
    embedding = relationship("Embedding", uselist=False, cascade="all, delete-orphan", back_populates="chunk")

    __table_args__ = (
        Index("ix_chunks_doc_idx", "doc_id", "chunk_index"),
    )

class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(Integer, ForeignKey("chunks.chunk_id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    model = Column(String(128), nullable=False)
    dim = Column(Integer, nullable=False)
    # Store raw vectors for provenance; ANN search is handled by Vertex AI Vector Search
    vector = Column(LargeBinary, nullable=False)

    chunk = relationship("Chunk", back_populates="embedding")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def init_db():
    """Create tables if absent and apply idempotent lightweight migrations.

    Ensures new nullable metadata columns (doc_type, jurisdiction) exist. Uses
    explicit transaction with engine.begin() so DDL is committed (previous
    implementation used connect() which could leave DDL uncommitted leading to
    runtime 'column does not exist' errors).
    """
    engine = _ensure_engine()
    Base.metadata.create_all(engine)

    try:
        with engine.begin() as conn:  # begin() ensures commit
            # Add columns if they don't exist (Postgres 9.6+ supports IF NOT EXISTS)
            conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_type VARCHAR(128) NULL"))
            conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS jurisdiction VARCHAR(64) NULL"))
        _logger.info("Database migration check complete: doc_type & jurisdiction ensured on documents table")
    except Exception as e:  # pragma: no cover
        _logger.warning("Migration check failed (continuing): %s", e)

def get_session():
    _ensure_engine()
    if _SESSION_FACTORY is None:
        raise RuntimeError("Session factory not initialized")
    return _SESSION_FACTORY()

# Flask helper (keeps parity with prior get_db using g)
def get_db():
    if not hasattr(g, "db_session"):
        g.db_session = get_session()
    return g.db_session

@contextlib.contextmanager
def session_scope():
    """Provide a transactional scope."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# --------------- CRUD ---------------
def add_document(*, sha256, title=None, source_path=None, doc_type=None, jurisdiction=None):
    """Insert a document if absent; return Document.

    If the document already exists, optionally update metadata fields if they are currently null
    and new values are supplied (non-breaking enrichment behavior).
    """
    with session_scope() as s:
        doc = s.query(Document).filter_by(sha256=sha256).one_or_none()
        if doc:
            updated = False
            if doc_type and not getattr(doc, "doc_type", None):
                doc.doc_type = doc_type
                updated = True
            if jurisdiction and not getattr(doc, "jurisdiction", None):
                doc.jurisdiction = jurisdiction
                updated = True
            if updated:
                s.flush()
            return doc
        doc = Document(sha256=sha256, title=title, source_path=source_path, doc_type=doc_type, jurisdiction=jurisdiction)
        s.add(doc)
        s.flush()
        return doc

def add_chunk(doc_id: int, text: str, chunk_index: int, token_count: int | None = None,
              page_start: int | None = None, page_end: int | None = None, section: str | None = None,
              tok_ver: int = TOK_VER, seg_ver: int = SEG_VER):
    with session_scope() as s:
        ch = Chunk(doc_id=doc_id, text=text, chunk_index=chunk_index,
                   token_count=token_count, tok_ver=tok_ver, seg_ver=seg_ver,
                   page_start=page_start, page_end=page_end, section=section)
        s.add(ch)
        s.flush()
        return ch

def upsert_embedding(chunk_id: int, model: str, dim: int, vector_bytes: bytes):
    with session_scope() as s:
        emb = s.query(Embedding).filter_by(chunk_id=chunk_id).one_or_none()
        if emb:
            # Update the existing embedding
            s.query(Embedding).filter_by(chunk_id=chunk_id).update({
                'model': model,
                'dim': dim,
                'vector': vector_bytes
            })
            s.flush()
            return s.query(Embedding).filter_by(chunk_id=chunk_id).one()
        emb = Embedding(chunk_id=chunk_id, model=model, dim=dim, vector=vector_bytes)
        s.add(emb)
        s.flush()
        return emb

def get_document(doc_id: int):
    with session_scope() as s:
        return s.query(Document).filter_by(doc_id=doc_id).one_or_none()

def get_document_by_sha(sha256: str):
    with session_scope() as s:
        return s.query(Document).filter_by(sha256=sha256).one_or_none()

def get_chunks_for_doc(doc_id: int):
    with session_scope() as s:
        return s.query(Chunk).filter_by(doc_id=doc_id).order_by(Chunk.chunk_index.asc()).all()

def get_chunks_by_ids(chunk_ids: list[int]):
    with session_scope() as s:
        # joinedload to pre-fetch related Document for filter display without extra queries
        return (
            s.query(Chunk)
            .options(joinedload(Chunk.document))
            .filter(Chunk.chunk_id.in_(chunk_ids))
            .all()
        )

def get_documents(limit: int = 100, offset: int = 0):
    """Return a slice of documents ordered by newest first."""
    with session_scope() as s:
        return (
            s.query(Document)
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

def get_document_chunk_counts(doc_ids: list[int]) -> dict[int, int]:
    """Return mapping doc_id -> chunk_count for supplied doc_ids."""
    if not doc_ids:
        return {}
    with session_scope() as s:
        rows = (
            s.query(Chunk.doc_id.label("doc_id"), func.count(Chunk.chunk_id).label("c"))
            .filter(Chunk.doc_id.in_(doc_ids))
            .group_by(Chunk.doc_id)
            .all()
        )
        out: dict[int, int] = {}
        for row in rows:
            did = int(getattr(row, "doc_id"))
            out[did] = int(getattr(row, "c"))
        return out

def get_all_chunks(limit: int = 200, offset: int = 0):
    """Return global slice of chunks ordered by chunk_id asc with related document eager loaded."""
    with session_scope() as s:
        return (
            s.query(Chunk)
            .options(joinedload(Chunk.document))
            .order_by(Chunk.chunk_id.asc())
            .offset(offset)
            .limit(limit)
            .all()
        )

def get_embedding_for_chunk(chunk_id: int):
    with session_scope() as s:
        return s.query(Embedding).filter_by(chunk_id=chunk_id).one_or_none()

def delete_document(doc_id: int):
    with session_scope() as s:
        doc = s.query(Document).filter_by(doc_id=doc_id).one_or_none()
        if doc:
            s.delete(doc)

def clear_database():
    with session_scope() as s:
        s.query(Embedding).delete()
        s.query(Chunk).delete()
        s.query(Document).delete()

# Convenience for health checks
def ping():
    engine = _ensure_engine()
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

__all__ = [
    "init_db", "get_session", "get_db", "session_scope",
    "add_document", "add_chunk", "upsert_embedding",
    "get_document", "get_document_by_sha", "get_chunks_for_doc", "get_chunks_by_ids",
    "get_documents", "get_document_chunk_counts", "get_all_chunks",
    "get_embedding_for_chunk", "delete_document", "clear_database",
    "TOK_VER", "SEG_VER", "ping",
    "Document", "Chunk", "Embedding",
]
