from __future__ import annotations

from typing import TypedDict

COL_FEATURE_ID = "feature_id"
COL_GEOMETRY = "geometry"
COL_CHUNK_KEY = "chunk_key"


class ChunkJob(TypedDict):
    chunk_id: tuple[int, ...]
    feature_ids: list[int | str]
