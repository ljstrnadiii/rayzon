from __future__ import annotations

from typing import TypedDict

import numpy as np

COL_FEATURE_ID = "feature_id"
COL_GEOMETRY = "geometry"
COL_CHUNK_KEY = "chunk_key"
COL_PIXELS = "pixels"


class PartialRow(TypedDict):
    feature_id: int | str
    dim_values: tuple[int, ...]
    pixels: np.ndarray


class ChunkJob(TypedDict):
    chunk_id: tuple[int, ...]
    feature_ids: list[int | str]
