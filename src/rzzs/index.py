from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import pyarrow as pa
from shapely import from_wkb
from shapely.geometry.base import BaseGeometry

from rzzs.grid import GridSpec, world_bbox_to_chunk_ranges
from rzzs.types import ChunkId, ChunkJob, FeatureId

GroupedRow: TypeAlias = dict[str, str | list[FeatureId]]
BBoxLike: TypeAlias = Mapping[str, float] | Sequence[float]
GeometryLike: TypeAlias = BaseGeometry | bytes | bytearray | memoryview


@dataclass(frozen=True)
class ChunkFeatureIndex:
    feature_to_chunks: dict[FeatureId, list[ChunkId]]
    chunk_to_features: dict[ChunkId, list[FeatureId]]


def build_feature_chunk_index_from_chunk_to_features(
    chunk_to_features: dict[ChunkId, list[FeatureId]],
) -> ChunkFeatureIndex:
    feature_to_chunks = _build_feature_to_chunks(chunk_to_features)

    return ChunkFeatureIndex(
        feature_to_chunks=feature_to_chunks,
        chunk_to_features=chunk_to_features,
    )


def plan_chunk_jobs(index: ChunkFeatureIndex) -> list[ChunkJob]:
    jobs: list[ChunkJob] = []
    for chunk_id, feature_ids in index.chunk_to_features.items():
        jobs.append({"chunk_id": chunk_id, "feature_ids": feature_ids})
    jobs.sort(key=lambda item: item["chunk_id"])
    return jobs


def _build_chunk_id_for_xy(chunk_y: int, chunk_x: int, grid: GridSpec) -> ChunkId:
    ids = [0 for _ in grid.dims]
    ids[grid.y_index] = chunk_y
    ids[grid.x_index] = chunk_x
    return tuple(ids)


def build_chunk_to_features_from_grouped_rows(
    grouped_rows: list[GroupedRow],
) -> dict[ChunkId, list[FeatureId]]:
    chunk_to_features: dict[ChunkId, list[FeatureId]] = {}
    for row in grouped_rows:
        chunk_key = row["chunk_key"]
        raw_feature_ids = row["feature_ids"]
        if not isinstance(chunk_key, str):
            raise TypeError("chunk_key values must be strings")
        if not isinstance(raw_feature_ids, list):
            raise TypeError("feature_ids values must be lists")
        chunk_id = _chunk_key_to_chunk_id(chunk_key)
        chunk_to_features[chunk_id] = [_normalize_feature_id(v) for v in raw_feature_ids]

    return chunk_to_features


def _map_feature_batch_to_chunk_rows(
    batch: pa.Table,
    *,
    grid_spec: GridSpec,
    feature_id_col: str,
) -> pa.Table:
    chunk_keys: list[str] = []
    feature_ids: list[FeatureId] = []

    for feature_id, minx, miny, maxx, maxy in _iter_normalized_feature_rows_from_arrow(
        _coerce_arrow_table(batch),
        feature_id_col=feature_id_col,
    ):
        y_range, x_range = world_bbox_to_chunk_ranges((minx, miny, maxx, maxy), grid_spec)
        for cy in y_range:
            for cx in x_range:
                chunk_id = _build_chunk_id_for_xy(cy, cx, grid_spec)
                chunk_keys.append(_chunk_id_to_key(chunk_id))
                feature_ids.append(feature_id)

    return pa.table({"chunk_key": chunk_keys, "feature_id": feature_ids})


def _group_chunk_feature_rows(batch: pa.Table) -> pa.Table:
    table = _coerce_arrow_table(batch)
    if table.num_rows == 0:
        return pa.table({"chunk_key": [], "feature_ids": []})

    chunk_key = table.column("chunk_key")[0].as_py()
    if not isinstance(chunk_key, str):
        raise TypeError("chunk_key values must be strings")

    feature_ids = [_normalize_feature_id(v) for v in table.column("feature_id").to_pylist()]
    return pa.table({"chunk_key": [chunk_key], "feature_ids": [feature_ids]})


def _iter_normalized_feature_rows_from_arrow(
    table: pa.Table,
    *,
    feature_id_col: str,
) -> Iterator[tuple[FeatureId, float, float, float, float]]:
    columns = set(table.column_names)
    if feature_id_col not in columns:
        raise KeyError(f"Missing required feature id column: {feature_id_col}")

    feature_ids = table.column(feature_id_col).to_pylist()
    required_bbox_cols = ("xmin", "ymin", "xmax", "ymax")

    if all(col in columns for col in required_bbox_cols):
        xmin_vals = table.column("xmin").to_pylist()
        ymin_vals = table.column("ymin").to_pylist()
        xmax_vals = table.column("xmax").to_pylist()
        ymax_vals = table.column("ymax").to_pylist()
        for feature_id, xmin, ymin, xmax, ymax in zip(
            feature_ids,
            xmin_vals,
            ymin_vals,
            xmax_vals,
            ymax_vals,
            strict=True,
        ):
            yield (
                _normalize_feature_id(feature_id),
                float(xmin),
                float(ymin),
                float(xmax),
                float(ymax),
            )
        return

    if "bbox" in columns:
        bbox_vals = table.column("bbox").to_pylist()
        for feature_id, bbox in zip(feature_ids, bbox_vals, strict=True):
            xmin, ymin, xmax, ymax = _bbox_to_tuple(bbox)
            yield (_normalize_feature_id(feature_id), xmin, ymin, xmax, ymax)
        return

    if "geometry" in columns:
        geometry_vals = table.column("geometry").to_pylist()
        for feature_id, geometry in zip(feature_ids, geometry_vals, strict=True):
            xmin, ymin, xmax, ymax = _normalize_geometry_value(geometry).bounds
            yield (_normalize_feature_id(feature_id), xmin, ymin, xmax, ymax)
        return

    raise KeyError("Feature input must include bbox columns, bbox struct, or geometry")


def _coerce_arrow_table(batch: pa.Table | pa.RecordBatch) -> pa.Table:
    if isinstance(batch, pa.Table):
        return batch
    if isinstance(batch, pa.RecordBatch):
        return pa.Table.from_batches([batch])
    raise TypeError(f"Unsupported Ray batch type: {type(batch)!r}")


def _build_feature_to_chunks(
    chunk_to_features: dict[ChunkId, list[FeatureId]],
) -> dict[FeatureId, list[ChunkId]]:
    feature_to_chunks: dict[FeatureId, list[ChunkId]] = defaultdict(list)
    for chunk_id, feature_ids in chunk_to_features.items():
        for feature_id in feature_ids:
            feature_to_chunks[feature_id].append(chunk_id)
    return {feature_id: chunk_ids for feature_id, chunk_ids in feature_to_chunks.items()}


def _chunk_id_to_key(chunk_id: ChunkId) -> str:
    return ",".join(str(part) for part in chunk_id)


def _chunk_key_to_chunk_id(chunk_key: str) -> ChunkId:
    if chunk_key == "":
        raise ValueError("chunk_key must not be empty")
    return tuple(int(part) for part in chunk_key.split(","))


def _normalize_feature_id(value: FeatureId) -> FeatureId:
    if isinstance(value, int | str):
        return value
    raise TypeError("feature_id values must be int or str")


def _bbox_to_tuple(value: BBoxLike) -> tuple[float, float, float, float]:
    if isinstance(value, Mapping):
        return (
            float(value["xmin"]),
            float(value["ymin"]),
            float(value["xmax"]),
            float(value["ymax"]),
        )
    if isinstance(value, list | tuple) and len(value) == 4:
        xmin, ymin, xmax, ymax = value
        return (float(xmin), float(ymin), float(xmax), float(ymax))
    raise TypeError("bbox values must be dict-like with xmin/ymin/xmax/ymax")


def _normalize_geometry_value(value: GeometryLike) -> BaseGeometry:
    if isinstance(value, BaseGeometry):
        return value
    if isinstance(value, bytes | bytearray | memoryview):
        return from_wkb(bytes(value))
    raise TypeError("geometry values must be shapely geometry or WKB bytes")
