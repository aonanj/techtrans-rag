from dotenv import load_dotenv
import os
from flask import current_app, has_app_context

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    CHAT_MODEL = os.getenv("CHAT_MODEL")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CLAUDE_KEY = os.getenv("CLAUDE_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    MAX_CONTEXT_LENGTH = os.getenv("MAX_CONTEXT_LENGTH")
    TOP_K_RESULTS = os.getenv("TOP_K_RESULTS")
    MAX_QUERY_LENGTH = os.getenv("MAX_QUERY_LENGTH")
    REQUEST_TIMEOUT = os.getenv("REQUEST_TIMEOUT")
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "gs://tx-rag-corpus/corpus_raw")
    CLEAN_FOLDER = os.getenv("CLEAN_FOLDER", "gs://tx-rag-corpus/corpus_clean")
    METADATA_FOLDER = os.getenv("METADATA_FOLDER", "gs://tx-rag-corpus/manifest")
    CHUNKS_FOLDER = os.getenv("CHUNKS_FOLDER", "gs://tx-rag-corpus/chunks")
    DB_PATH = os.getenv("DB_PATH", "gs://tx-rag-corpus/app.db")
    DB_NAME = os.getenv("DB_NAME", "appdb")
    DB_USER = os.getenv("DB_USER", "appuser")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "appuser")
    DB_HOST = os.getenv("DB_HOST", "10.25.1.3")
    DB_PORT = os.getenv("DB_PORT", "5432")
    CLOUD_SQL_CONNECTION_NAME = os.getenv("CLOUD_SQL_CONNECTION_NAME", "tech-trans-rag:us-west2:tx-rag-pg")
    DB_IAM_AUTH=os.getenv("DB_IAM_AUTH", "False").lower() in ("true", "1", "t")
    DB_USE_CONNECTOR=os.getenv("DB_USE_CONNECTOR", "False").lower() in ("true", "1", "t")
    VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "tech-trans-rag")
    VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-west2")
    VERTEX_INDEX_ENDPOINT_ID = os.getenv("VERTEX_INDEX_ENDPOINT_ID", "6972425444192681984")
    VERTEX_INDEX_ID = os.getenv("VERTEX_INDEX_ID", "9085431706080509952")
    VERTEX_DEPLOYED_INDEX_ID = os.getenv("VERTEX_DEPLOYED_INDEX_ID", "chunks_deployed")

def cfg(key, default=None, cast=lambda x: x):
    val = current_app.config.get(key) if has_app_context() else None
    if val is None:
        val = os.getenv(key, default)
    return cast(val) if val is not None else val