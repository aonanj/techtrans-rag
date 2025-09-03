import sqlite3
from pathlib import Path
from flask import current_app, g
from infrastructure.document_processor import sha256_text

TOK_VER = 1 # increment if tokenization changes
SEG_VER = 1 # increment if segmentation changes

def get_db():
    if not hasattr(g, "db"):
        db_path = current_app.config.get("DB_PATH", "data/app.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        g.db = conn
    return g.db

def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    db.execute(
        """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                title TEXT,
                file_hash TEXT NOT NULL,
                modified_at INTEGER NOT NULL
            );
        """
    )
    db.execute(
        """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                page_start INTEGER,
                page_end INTEGER,
                section TEXT,
                chunk_hash TEXT NOT NULL,
                content_hash TEXT NOT NULL,     
                text TEXT NOT NULL,
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
            );
        """
    )
    db.execute(
        """
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL,
                FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
            );
        """
    )
    db.execute(
        """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_doc_hash ON chunks(doc_id, chunk_hash);
        """
    )
    db.execute(
        """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_chunk_model ON embeddings(chunk_id, model);
        """
    )
    db.commit()

def upsert_document(db, doc_id: str, source_path: str, title: str, norm_text: str, mtime: float) -> bool:
    file_hash = sha256_text(norm_text)
    cur = db.execute("SELECT file_hash FROM documents WHERE doc_id=?", (doc_id,))
    row = cur.fetchone()
    if row and row["file_hash"] == file_hash:
        return False  # no change
    db.execute("REPLACE INTO documents(doc_id, source_path, title, file_hash, modified_at) VALUES (?,?,?,?,?)",
               (doc_id, source_path, title, file_hash, int(mtime)))
    db.commit()
    return True 

def persist_chunk(db, doc_id, text, page_s, page_e, section):
    chunk_hash = sha256_text(text)
    content_hash = sha256_text(f"{text}|tok={TOK_VER}|seg={SEG_VER}")
    chunk_id = f"{doc_id}:{chunk_hash[:12]}"
    db.execute("""REPLACE INTO chunks(chunk_id, doc_id, page_start, page_end, section, chunk_hash, content_hash, text)
               VALUES (?,?,?,?,?,?,?,?)""",
               (chunk_id, doc_id, page_s, page_e, section, chunk_hash, content_hash, text))
    db.commit()
    return chunk_id, content_hash


def print_all_documents(db=None):
    """Print all rows from the documents table to stdout and return them.

    If no connection is passed, obtains one via get_db(). Returns the list of
    sqlite3.Row objects (may be empty). Swallows errors if table is missing.
    """
    if db is None:
        db = get_db()
    try:
        cur = db.execute("SELECT doc_id, source_path, title, file_hash, modified_at FROM documents ORDER BY modified_at DESC")
        rows = cur.fetchall()
        for r in rows:
            # Convert Row to regular dict for a cleaner print representation
            print({k: r[k] for k in r.keys()})
        return rows
    except Exception as e:
        print(f"Error printing documents: {e}")
        return []

def delete_document(doc_id: str):
    """
    Delete a document and its related chunks/embeddings.
    Foreign keys were declared but no ON DELETE CASCADE, so delete manually.
    """
    db = get_db()
    # Ensure FK enforcement (SQLite off by default per-connection)
    db.execute("PRAGMA foreign_keys=ON;")
    # Delete embeddings referencing chunks of this doc
    db.execute("""DELETE FROM embeddings
                  WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id=?)""",
               (doc_id,))
    # Delete chunks
    db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    # Delete document
    db.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
    db.commit()

def clear_database():
    db = get_db()
    db.execute("DELETE FROM documents")
    db.execute("DELETE FROM chunks")
    db.execute("DELETE FROM embeddings")
    db.commit()
