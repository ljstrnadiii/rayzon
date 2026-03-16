from __future__ import annotations

from typing import TypeAlias, TypedDict

import numpy as np
from affine import Affine
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

ScalarValue: TypeAlias = int | float | str
FeatureId: TypeAlias = int | str
ChunkId: TypeAlias = tuple[int, ...]
DimValues: TypeAlias = tuple[ScalarValue, ...]
AffineTuple: TypeAlias = tuple[float, float, float, float, float, float]
AffineLike: TypeAlias = Affine | AffineTuple | list[float] | NDArray[np.floating]
GeometryStore: TypeAlias = dict[FeatureId, BaseGeometry]


class MosaicHandle(TypedDict):
    store_uri: str
    array_path: str


class PartialRow(TypedDict, total=False):
    feature_id: FeatureId
    dim_values: DimValues
    stat_keys: tuple[str, ...]
    count: int
    n_valid: int
    sum: float
    sum_sq: float
    min: float
    max: float
    tdigest_centroids_mean: np.ndarray | None
    tdigest_centroids_weight: np.ndarray | None


class FinalRow(TypedDict, total=False):
    feature_id: FeatureId
    count: int
    n_valid: int
    sum: float
    mean: float
    variance: float
    std: float
    min: float
    max: float


class ChunkJob(TypedDict):
    chunk_id: ChunkId
    feature_ids: list[FeatureId]
