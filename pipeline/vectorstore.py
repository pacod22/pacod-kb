"""Vector store abstraction: Qdrant embedded (in-memory / local path) and server."""

from __future__ import annotations

import hashlib
import uuid as _uuid
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pipeline.chunker import Chunk
    from pipeline.config import Config


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Deterministic UUID from chunk_id string (MD5 → UUID bytes)."""
    return str(_uuid.UUID(bytes=hashlib.md5(chunk_id.encode()).digest()))


ReuseKey = tuple[str, str]


def reuse_key(content_hash: str, embedded_text: str) -> ReuseKey:
    """Identity of an embedding: the chunk's hash plus a digest of the exact embedded text.

    content_hash covers the raw chunk only; the breadcrumb (document title, headings) is
    part of what was embedded, so it must match too.
    """
    return content_hash, hashlib.sha1(embedded_text.encode()).hexdigest()


class VectorStore(Protocol):
    """Protocol for vector store backends."""

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        doc_meta: dict[str, Any] | None = None,
    ) -> None: ...

    def query(
        self,
        vector: list[float],
        top_k: int,
        *,
        include_historical: bool = False,
        valid_until_in: Sequence[str] | None = None,
    ) -> list[tuple[str, int]]:
        """Return [(chunk_id, rank)] sorted best-first (rank starts at 1).

        *valid_until_in* keeps current chunks plus those whose valid_until is one of
        the given dates (the as_of pushdown); it overrides *include_historical*.
        """
        ...

    def delete_by_doc(self, doc_id: str) -> None: ...

    def set_valid_until(self, doc_id: str, valid_until: str | None) -> None: ...

    def retag_project(self, old: str, new: str) -> None: ...

    def drop_all(self) -> None: ...

    @property
    def client(self) -> Any: ...

    @property
    def collection_name(self) -> str: ...


_VECTOR_SIZE = 1024  # BGE-M3 dense dimension
_UPSERT_BATCH = 256


class QdrantVectorStore:
    """Wraps a QdrantClient for chunk storage and retrieval."""

    def __init__(self, client: Any, collection_name: str, *, ensure: bool = True) -> None:
        """*ensure* creates the collection and its payload indexes when missing.

        Pass False for a read-only handle on a collection known to exist (saves the
        existence and schema round-trips per request).
        """
        self._client = client
        self._collection_name = collection_name
        if ensure:
            self._ensure_collection()

    def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not self._client.collection_exists(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
            )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        """Create missing payload indexes (server mode; local Qdrant ignores them).

        Runs for existing collections too, so indexes added later (valid_until is
        filtered on by every current-only query) reach older collections.
        """
        from qdrant_client.models import PayloadSchemaType

        if type(getattr(self._client, "_client", None)).__name__ == "QdrantLocal":
            return
        info = self._client.get_collection(self._collection_name)
        present = set((info.payload_schema or {}).keys())
        for field, schema in (
            ("source_id", PayloadSchemaType.KEYWORD),
            ("source_type", PayloadSchemaType.KEYWORD),
            ("language", PayloadSchemaType.KEYWORD),
            ("legal_year", PayloadSchemaType.INTEGER),
            ("valid_until", PayloadSchemaType.KEYWORD),
            ("project", PayloadSchemaType.KEYWORD),
            ("content_hash", PayloadSchemaType.KEYWORD),  # embedding reuse lookups
        ):
            if field not in present:
                self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=field,
                    field_schema=schema,
                )

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        doc_meta: dict[str, Any] | None = None,
    ) -> None:
        from qdrant_client.models import PointStruct

        meta = doc_meta or {}
        points = [
            PointStruct(
                id=_chunk_id_to_uuid(c.chunk_id),
                vector=v,
                payload={
                    "chunk_id": c.chunk_id,
                    "source_id": c.doc_id,
                    "heading_path": c.heading_path,
                    "heading_level": c.heading_level,
                    "parent_id": c.parent_chunk,
                    "sequence": c.sequence,
                    "byte_start": c.byte_start,
                    "byte_end": c.byte_end,
                    "valid_until": c.valid_until,
                    "extraction_method": c.extraction_method,
                    "confidence": c.confidence,
                    "text": c.embedded_text,
                    "raw_text": c.text,
                    "chunk_type": "child",
                    "content_hash": c.content_hash,
                    # Which model made the vector: reuse never mixes models.
                    "embedding_model": meta.get("embedding_model"),
                    "title": meta.get("title", ""),
                    "source_type": meta.get("source_type", "generic"),
                    "legal_year": meta.get("legal_year", 0),
                    "language": meta.get("language", "unknown"),
                    "url": meta.get("url"),
                    "project": meta.get("project"),
                    "retrieved_at": meta.get("retrieved_at"),
                    "converted_at": meta.get("converted_at"),
                },
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        # Batched: one request with thousands of 1024-d vectors plus text
        # payloads can exceed the server's request size limit and time out.
        for start in range(0, len(points), _UPSERT_BATCH):
            self._client.upsert(
                collection_name=self._collection_name,
                points=points[start : start + _UPSERT_BATCH],
            )

    def query(
        self,
        vector: list[float],
        top_k: int,
        *,
        include_historical: bool = False,
        valid_until_in: Sequence[str] | None = None,
    ) -> list[tuple[str, int]]:
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            IsNullCondition,
            MatchAny,
            PayloadField,
        )

        current = IsNullCondition(is_null=PayloadField(key="valid_until"))
        query_filter: Filter | None = None
        if valid_until_in is not None:
            # A keyword match, not DatetimeRange: embedded Qdrant evaluates date
            # ranges per point in Python (parsing every payload date each query).
            conditions: list[Any] = [current]
            if valid_until_in:
                conditions.append(
                    FieldCondition(key="valid_until", match=MatchAny(any=list(valid_until_in)))
                )
            query_filter = Filter(should=conditions)
        elif not include_historical:
            query_filter = Filter(must=[current])

        results = self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            # Only the id is used: text and metadata are hydrated from the keyword store.
            with_payload=["chunk_id"],
        )
        return [(p.payload["chunk_id"], i + 1) for i, p in enumerate(results.points)]

    def _reusable(
        self,
        query_filter: Any,
        model: str,
        allow_untagged: bool,
        wanted: set[ReuseKey] | None,
        limit: int,
    ) -> dict[ReuseKey, Any]:
        """Scroll points, keeping vectors whose (hash, text) is wanted and model matches."""
        out: dict[ReuseKey, Any] = {}
        offset: Any = None
        while len(out) < limit:
            points, offset = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=query_filter,
                limit=min(1000, limit),
                offset=offset,
                with_vectors=True,
                with_payload=["content_hash", "text", "embedding_model"],
            )
            for pt in points:
                payload = pt.payload or {}
                tag = payload.get("embedding_model")
                if not (tag == model or (tag is None and allow_untagged)):
                    continue
                key = reuse_key(
                    str(payload.get("content_hash") or ""), str(payload.get("text") or "")
                )
                if (wanted is None or key in wanted) and pt.vector is not None:
                    out[key] = pt.vector
            if offset is None:
                break
        return out

    def fetch_reusable_vectors(
        self,
        keys: Sequence[tuple[str, str]],
        *,
        model: str,
        allow_untagged: bool = False,
    ) -> dict[ReuseKey, list[float]]:
        """Stored vectors for chunks given as (content_hash, embedded_text) pairs.

        Superseding a document re-chunks it under a new doc_id; unchanged sections keep
        their hash and text, so their vectors come from the previous version's points.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        wanted = {reuse_key(h, t) for h, t in keys if h}
        hashes = sorted({h for h, _ in wanted})
        out: dict[ReuseKey, list[float]] = {}
        for start in range(0, len(hashes), 256):
            part = hashes[start : start + 256]
            flt = Filter(must=[FieldCondition(key="content_hash", match=MatchAny(any=part))])
            found = self._reusable(flt, model, allow_untagged, wanted, limit=10 * len(part) + 100)
            out.update({k: list(v) for k, v in found.items()})
        return out

    def snapshot_vectors(
        self, *, model: str, allow_untagged: bool = False, limit: int = 100_000
    ) -> dict[ReuseKey, Any]:
        """Every reusable vector in the collection (float32 arrays), up to *limit*.

        Taken before a rebuild drops the collection, so unchanged chunks are not
        re-embedded.
        """
        import numpy as np

        found = self._reusable(None, model, allow_untagged, None, limit)
        return {k: np.asarray(v, dtype=np.float32) for k, v in found.items()}

    def delete_by_doc(self, doc_id: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="source_id", match=MatchValue(value=doc_id))]
            ),
        )

    def set_valid_until(self, doc_id: str, valid_until: str | None) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self._client.set_payload(
            collection_name=self._collection_name,
            payload={"valid_until": valid_until},
            points=Filter(
                must=[FieldCondition(key="source_id", match=MatchValue(value=doc_id))]
            ),
        )

    def retag_project(self, old: str, new: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self._client.set_payload(
            collection_name=self._collection_name,
            payload={"project": new},
            points=Filter(must=[FieldCondition(key="project", match=MatchValue(value=old))]),
        )

    def drop_all(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._ensure_collection()

    @property
    def client(self) -> Any:
        return self._client

    @property
    def collection_name(self) -> str:
        return self._collection_name


def make_vector_store(config: Config) -> QdrantVectorStore:
    """Build a QdrantVectorStore from config."""
    from qdrant_client import QdrantClient

    if config.qdrant_mode == "server":
        client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
        )
    else:
        # Embedded: use in-memory if index_dir doesn't exist yet, else local path
        qdrant_path = config.index_dir / "qdrant"
        qdrant_path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(qdrant_path))

    return QdrantVectorStore(client, config.qdrant_collection)
