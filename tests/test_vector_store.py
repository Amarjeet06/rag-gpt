"""
Real integration tests for backend/vector_store.py against a live Postgres +
pgvector instance. Requires DATABASE_URL to be set (CI provides this via a
postgres service container with the pgvector extension available - see
.github/workflows/ci.yml). Skipped automatically if DATABASE_URL is unset,
so local development without a database still works.
"""
import os
import random

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set - skipping live Postgres/pgvector tests",
)

from backend import vector_store  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_table():
    vector_store.init_db()
    session = vector_store._session()
    try:
        session.query(vector_store.DocumentChunk).delete()
        session.commit()
    finally:
        session.close()
    yield


def _rand_vec(dim=8, seed=None):
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(dim)]


def test_is_enabled_reflects_database_url():
    assert vector_store.is_enabled() is True


def test_replace_and_retrieve_roundtrip():
    chunks = ["alpha chunk", "beta chunk", "gamma chunk"]
    embeddings = [
        [1.0] + [0.0] * (vector_store.EMBEDDING_DIM - 1),
        [0.0, 1.0] + [0.0] * (vector_store.EMBEDDING_DIM - 2),
        [0.0, 0.0, 1.0] + [0.0] * (vector_store.EMBEDDING_DIM - 3),
    ]
    count = vector_store.replace_chunks("alice", "chat1", "doc.pdf", chunks, embeddings)
    assert count == 3
    assert vector_store.chunk_count("alice", "chat1") == 3


def test_retrieve_top_ranks_nearest_neighbor_first():
    dim = vector_store.EMBEDDING_DIM
    chunks = ["close match", "far match 1", "far match 2"]
    target = [1.0] + [0.0] * (dim - 1)
    embeddings = [
        target,                                   # identical to the query -> should rank #1
        [-1.0] + [0.0] * (dim - 1),                # opposite direction -> should rank last
        [0.0, 1.0] + [0.0] * (dim - 2),            # orthogonal -> should rank #2
    ]
    vector_store.replace_chunks("bob", "chat1", "doc.pdf", chunks, embeddings)

    results = vector_store.retrieve_top("bob", "chat1", target, k=3)
    assert len(results) == 3
    assert results[0]["text"] == "close match"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)
    assert results[-1]["text"] == "far match 1"
    assert results[-1]["score"] == pytest.approx(-1.0, abs=1e-4)


def test_replace_chunks_overwrites_previous_upload():
    dim = vector_store.EMBEDDING_DIM
    vector_store.replace_chunks("carol", "chat1", "v1.pdf", ["old chunk"], [[0.1] * dim])
    assert vector_store.chunk_count("carol", "chat1") == 1

    vector_store.replace_chunks("carol", "chat1", "v2.pdf", ["new chunk a", "new chunk b"], [[0.2] * dim, [0.3] * dim])
    assert vector_store.chunk_count("carol", "chat1") == 2


def test_chunks_are_isolated_per_user_and_chat():
    dim = vector_store.EMBEDDING_DIM
    vector_store.replace_chunks("dave", "chatA", "a.pdf", ["dave chunk"], [[0.5] * dim])
    vector_store.replace_chunks("erin", "chatA", "b.pdf", ["erin chunk"], [[0.5] * dim])

    dave_results = vector_store.retrieve_top("dave", "chatA", [0.5] * dim, k=5)
    assert len(dave_results) == 1
    assert dave_results[0]["text"] == "dave chunk"
