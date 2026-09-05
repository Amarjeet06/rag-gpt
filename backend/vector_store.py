"""
Persistent vector storage for RAG-GPT using PostgreSQL + pgvector.

The original implementation kept every uploaded PDF's chunk embeddings
inline inside a per-user JSON file and did brute-force numpy cosine
similarity in-process on every query. That works for a single-process demo
but doesn't survive a restart cleanly, doesn't scale past one process, and
isn't how retrieval is actually done in production RAG systems.

This module is additive: when DATABASE_URL is set, chunks are persisted to
Postgres and similarity search uses pgvector's native cosine-distance
operator (<=>) with an IVFFlat index, instead of loading every embedding
into a numpy array by hand. When DATABASE_URL is not set, the app keeps
working exactly as before (JSON-file storage) - see app.py's use of
`vector_store.is_enabled()`.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, Column, Integer, String, Text, text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

DATABASE_URL = os.getenv("DATABASE_URL")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))  # all-MiniLM-L6-v2 = 384

Base = declarative_base()
_engine = None
_SessionLocal = None


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, index=True, nullable=False)
    chat_id = Column(String, index=True, nullable=False)
    source = Column(String, nullable=False)
    section_idx = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)


def is_enabled() -> bool:
    return bool(DATABASE_URL)


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL is not set; vector_store is disabled")
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine)
    return _engine


def init_db() -> None:
    """Creates the pgvector extension (if missing), the table, and an
    IVFFlat cosine-distance index. Safe to call repeatedly."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx "
            "ON document_chunks USING ivfflat (embedding vector_cosine_ops) "
            "WITH (lists = 100)"
        ))


def _session():
    if _SessionLocal is None:
        get_engine()
    return _SessionLocal()


def replace_chunks(username: str, chat_id: str, source: str, chunks: List[str], embeddings: List[List[float]]) -> int:
    """Replaces all stored chunks for this (username, chat_id) with a new set
    (mirrors the JSON path's behaviour of overwriting doc_index per upload)."""
    session = _session()
    try:
        session.query(DocumentChunk).filter_by(username=username, chat_id=chat_id).delete()
        rows = [
            DocumentChunk(
                username=username,
                chat_id=chat_id,
                source=source,
                section_idx=i,
                text=ch,
                embedding=emb,
            )
            for i, (ch, emb) in enumerate(zip(chunks, embeddings))
        ]
        session.bulk_save_objects(rows)
        session.commit()
        return len(rows)
    finally:
        session.close()


def retrieve_top(username: str, chat_id: str, query_embedding: List[float], k: int = 8) -> List[Dict[str, Any]]:
    """Returns the top-k chunks by cosine similarity using pgvector's native
    <=> (cosine distance) operator, so the ranking happens in the database
    rather than by loading every embedding into a Python list."""
    session = _session()
    try:
        rows = (
            session.query(
                DocumentChunk.text,
                DocumentChunk.source,
                DocumentChunk.section_idx,
                DocumentChunk.embedding.cosine_distance(query_embedding).label("distance"),
            )
            .filter_by(username=username, chat_id=chat_id)
            .order_by("distance")
            .limit(k)
            .all()
        )
        results = []
        for rank, row in enumerate(rows):
            similarity = 1.0 - float(row.distance)  # cosine_distance = 1 - cosine_similarity
            results.append({
                "rank": rank + 1,
                "score": similarity,
                "text": row.text,
                "source": row.source,
                "section_idx": row.section_idx,
            })
        return results
    finally:
        session.close()


def chunk_count(username: str, chat_id: Optional[str] = None) -> int:
    session = _session()
    try:
        q = session.query(DocumentChunk).filter_by(username=username)
        if chat_id is not None:
            q = q.filter_by(chat_id=chat_id)
        return q.count()
    finally:
        session.close()
