"""Shared FastAPI dependencies for the UI."""

from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import Depends, Request
from qdrant_client import QdrantClient


def get_pipeline_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.pipeline_client  # type: ignore[no-any-return]


def get_qdrant_client(request: Request) -> QdrantClient:
    client: QdrantClient | None = request.app.state.qdrant_client
    if client is None:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503, detail="Qdrant management unavailable in embedded mode"
        )
    return client


PipelineClientDep = Annotated[httpx.AsyncClient, Depends(get_pipeline_client)]
QdrantClientDep = Annotated[QdrantClient, Depends(get_qdrant_client)]
