from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from time import perf_counter

import pyarrow as pa

from rayzon.arrow import as_table, coerce_geometry
from rayzon.grid import GridSpec, reconstruct_grid_spec, world_bbox_to_chunk_ranges
from rayzon.logging_utils import ensure_basic_logging, get_logger
from rayzon.types import COL_CHUNK_KEY, COL_FEATURE_ID, COL_GEOMETRY, ChunkJob

logger = get_logger(__name__)


@dataclass(frozen=True)
class ChunkFeatureIndex:
    feature_to_chunks: dict[int | str, list[tuple[int, ...]]]
    chunk_to_features: dict[tuple[int, ...], list[int | str]]


def build_feature_chunk_index_from_chunk_to_features(
    chunk_to_features: dict[tuple[int, ...], list[int | str]],
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


def _build_chunk_ids_for_xy(
    chunk_y: int,
    chunk_x: int,
    grid: GridSpec,
    *,
    allowed_chunk_ids_by_dim: Mapping[str, list[int]] | None = None,
) -> list[tuple[int, ...]]:
    non_spatial_axes = [i for i in range(len(grid.dims)) if i != grid.y_index and i != grid.x_index]
    non_spatial_ranges: list[Sequence[int]] = []
    for ax in non_spatial_axes:
        dim = grid.dims[ax]
        allowed = (
            allowed_chunk_ids_by_dim.get(dim) if allowed_chunk_ids_by_dim is not None else None
        )
        if allowed is None:
            non_spatial_ranges.append(range(math.ceil(grid.shape[ax] / grid.chunk_sizes[ax])))
        else:
            non_spatial_ranges.append(tuple(allowed))
    if not non_spatial_ranges:
        ids = [0] * len(grid.dims)
        ids[grid.y_index] = chunk_y
        ids[grid.x_index] = chunk_x
        return [tuple(ids)]

    results: list[tuple[int, ...]] = []
    for combo in product(*non_spatial_ranges):
        ids = [0] * len(grid.dims)
        ids[grid.y_index] = chunk_y
        ids[grid.x_index] = chunk_x
        for ax, val in zip(non_spatial_axes, combo, strict=True):
            ids[ax] = val
        results.append(tuple(ids))
    return results


def _build_feature_to_chunks(
    chunk_to_features: dict[tuple[int, ...], list[int | str]],
) -> dict[int | str, list[tuple[int, ...]]]:
    feature_to_chunks: dict[int | str, list[tuple[int, ...]]] = defaultdict(list)
    for chunk_id, feature_ids in chunk_to_features.items():
        for feature_id in feature_ids:
            feature_to_chunks[feature_id].append(chunk_id)
    return dict(feature_to_chunks)


def _chunk_id_to_key(chunk_id: tuple[int, ...]) -> str:
    return ",".join(str(part) for part in chunk_id)


def _chunk_key_to_chunk_id(chunk_key: str) -> tuple[int, ...]:
    if chunk_key == "":
        raise ValueError("chunk_key must not be empty")
    return tuple(int(part) for part in chunk_key.split(","))


def _normalize_feature_id(value: int | str) -> int | str:
    if isinstance(value, int | str):
        return value
    raise TypeError("feature_id values must be int or str")


def map_feature_to_chunk_rows(
    batch: pa.Table,
    *,
    feature_id_col: str,
    geometry_col: str,
    dims: list[str],
    shape: list[int],
    chunk_sizes: list[int],
    transform_coeffs: list[float],
    crs: str,
    x_dim: str,
    y_dim: str,
    allowed_chunk_ids_by_dim: Mapping[str, list[int]] | None = None,
) -> pa.Table:
    ensure_basic_logging()
    started_at = perf_counter()
    grid_spec = reconstruct_grid_spec(
        dims=dims,
        shape=shape,
        chunk_sizes=chunk_sizes,
        transform_coeffs=transform_coeffs,
        crs=crs,
        x_dim=x_dim,
        y_dim=y_dim,
    )
    table = as_table(batch)

    feature_ids = table.column(feature_id_col).to_pylist()
    geom_vals = table.column(geometry_col).to_pylist()

    out_chunk_keys: list[str] = []
    out_feature_ids: list[int | str] = []
    out_geometries: list[bytes] = []
    total_candidate_spatial_chunks = 0

    for fid, geom_raw in zip(feature_ids, geom_vals, strict=True):
        feature_id = _normalize_feature_id(fid)
        geom = coerce_geometry(geom_raw)
        geom_wkb = geom.wkb
        minx, miny, maxx, maxy = geom.bounds

        y_range, x_range = world_bbox_to_chunk_ranges((minx, miny, maxx, maxy), grid_spec)
        total_candidate_spatial_chunks += len(y_range) * len(x_range)
        for cy in y_range:
            for cx in x_range:
                for chunk_id in _build_chunk_ids_for_xy(
                    cy,
                    cx,
                    grid_spec,
                    allowed_chunk_ids_by_dim=allowed_chunk_ids_by_dim,
                ):
                    out_chunk_keys.append(_chunk_id_to_key(chunk_id))
                    out_feature_ids.append(feature_id)
                    out_geometries.append(geom_wkb)

    out = pa.table(
        {
            COL_CHUNK_KEY: out_chunk_keys,
            COL_FEATURE_ID: out_feature_ids,
            COL_GEOMETRY: out_geometries,
        }
    )
    elapsed_s = perf_counter() - started_at
    unique_chunk_keys = len(set(out_chunk_keys))
    logger.info(
        "feature batch mapped input_rows=%d output_rows=%d unique_chunk_keys=%d "
        "candidate_spatial_chunks=%d avg_output_rows_per_feature=%.2f "
        "selected_chunk_dims=%s time_s=%.3f",
        table.num_rows,
        out.num_rows,
        unique_chunk_keys,
        total_candidate_spatial_chunks,
        (out.num_rows / table.num_rows) if table.num_rows else 0.0,
        sorted((allowed_chunk_ids_by_dim or {}).keys()),
        elapsed_s,
    )
    return out
