from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyproj
import ray.data
from affine import Affine
from ray.data.aggregate import AggregateFnV2

from rayzon.feature_dataset import (
    _detect_feature_id_type,
    _detect_num_blocks,
    to_feature_dataset,
)
from rayzon.grid import GridSpec
from rayzon.logging_utils import ensure_basic_logging, get_logger
from rayzon.rasterize_backend import RasterizeBackend
from rayzon.stats import (
    DEFAULT_STAT_EXPRS,
    ResolvedStats,
    build_aggregations,
    required_partial_columns,
    resolve_stat_exprs,
)
from rayzon.types import COL_FEATURE_ID
from rayzon.zarr_backend import ZarrBackend, build_grid_spec, resolve_dim_coords

if TYPE_CHECKING:
    import geopandas as gpd


logger = get_logger(__name__)


@dataclass(frozen=True)
class ZonalStatsPlan:
    resolved: ResolvedStats
    partial_columns: tuple[str, ...]
    grid_spec: GridSpec
    non_spatial_dims: list[str]
    requested_selectors: dict[str, object]
    dim_coord_values: dict[str, list]
    allowed_dim_indices: dict[str, list[int]]
    allowed_chunk_ids_by_dim: dict[str, list[int]]
    feature_ds: ray.data.Dataset
    feature_id_pa_type: pa.DataType
    feature_num_blocks: int | None
    feature_map_kwargs: dict[str, object]
    chunk_kwargs: dict[str, object]
    group_keys: list[str]
    group_key_pa_types: list[pa.DataType]
    aggs: list[AggregateFnV2[Any, Any]]
    plan_elapsed_s: float
    scale_factor: float | None
    add_offset: float | None


def build_zonal_stats_plan(
    store_uri: str,
    features: gpd.GeoDataFrame | str | Path | ray.data.Dataset,
    *,
    transform: Affine | None = None,
    crs: pyproj.CRS | None = None,
    x_dim: str = "x",
    y_dim: str = "y",
    stats: Sequence[str] = DEFAULT_STAT_EXPRS,
    array_name: str | None = None,
    all_touched: bool = False,
    nodata: float | int | None = None,
    decode_coords: bool = False,
    storage_options: Mapping[str, object] | None = None,
    selectors: Mapping[str, object] | None = None,
    feature_override_num_blocks: int | None = None,
    zarr_backend: ZarrBackend = ZarrBackend.ZARR_PYTHON,
    rasterize_backend: RasterizeBackend = RasterizeBackend.RASTERIO,
) -> ZonalStatsPlan:
    ensure_basic_logging()
    planning_started_at = perf_counter()
    resolved = resolve_stat_exprs(list(stats))
    partial_columns = required_partial_columns(resolved)
    transform_tuple = (
        (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
        if transform is not None
        else None
    )
    crs_wkt = crs.to_wkt() if crs is not None else None

    grid_spec, _, scale_factor, add_offset = build_grid_spec(
        store_uri,
        x_dim=x_dim,
        y_dim=y_dim,
        transform=transform_tuple,
        crs=crs_wkt,
        array_name=array_name,
        zarr_backend=zarr_backend,
        storage_options=storage_options,
    )
    non_spatial_dims = [dim for dim in grid_spec.dims if dim not in (y_dim, x_dim)]
    resolved_crs = pyproj.CRS(grid_spec.crs)
    requested_selectors = dict(selectors or {})
    invalid_selector_dims = [dim for dim in requested_selectors if dim not in non_spatial_dims]
    if invalid_selector_dims:
        raise ValueError(
            "selectors must target non-spatial dimensions only. "
            f"Got invalid values: {invalid_selector_dims!r}; "
            f"available dimensions: {non_spatial_dims!r}."
        )

    dim_coord_values: dict[str, list] = {}
    if (decode_coords or requested_selectors) and non_spatial_dims:
        dim_coord_values = resolve_dim_coords(
            store_uri,
            dims=grid_spec.dims,
            x_dim=x_dim,
            y_dim=y_dim,
            array_name=array_name,
            storage_options=storage_options,
        )
    allowed_dim_indices = _resolve_allowed_dim_indices(
        grid_spec=grid_spec,
        non_spatial_dims=non_spatial_dims,
        dim_coord_values=dim_coord_values,
        selectors=requested_selectors,
    )
    allowed_chunk_ids_by_dim = _resolve_allowed_chunk_ids_by_dim(
        grid_spec=grid_spec,
        allowed_dim_indices=allowed_dim_indices,
    )

    _check_feature_crs(features, resolved_crs)
    feature_ds = to_feature_dataset(features, override_num_blocks=feature_override_num_blocks)
    feature_id_pa_type = _detect_feature_id_type(feature_ds)
    feature_num_blocks = _detect_num_blocks(feature_ds)
    plan_elapsed_s = perf_counter() - planning_started_at

    grid_kwargs: dict[str, object] = {
        "dims": list(grid_spec.dims),
        "shape": list(grid_spec.shape),
        "chunk_sizes": list(grid_spec.chunk_sizes),
        "transform_coeffs": list(grid_spec.transform),
        "crs": grid_spec.crs,
        "x_dim": x_dim,
        "y_dim": y_dim,
    }
    feature_map_kwargs = {
        **grid_kwargs,
        "allowed_chunk_ids_by_dim": allowed_chunk_ids_by_dim,
    }
    chunk_kwargs = {
        "store_uri": store_uri,
        "array_name": array_name or "",
        "all_touched": all_touched,
        "nodata": nodata,
        "zarr_backend_value": zarr_backend.value,
        "rasterize_backend_value": rasterize_backend.value,
        "partial_columns": list(partial_columns),
        "feature_id_pa_type": str(feature_id_pa_type),
        "scale_factor": scale_factor,
        "add_offset": add_offset,
        "dim_coord_values": dim_coord_values,
        "allowed_dim_indices": allowed_dim_indices,
        "storage_options": dict(storage_options or {}),
        **grid_kwargs,
    }
    group_keys = [COL_FEATURE_ID] + non_spatial_dims
    group_key_pa_types = [feature_id_pa_type] + [
        _infer_group_key_pa_type(
            dim_coord_values.get(dim, []),
            decode_coords=decode_coords,
        )
        for dim in non_spatial_dims
    ]
    aggs = build_aggregations(resolved)
    return ZonalStatsPlan(
        resolved=resolved,
        partial_columns=partial_columns,
        grid_spec=grid_spec,
        non_spatial_dims=non_spatial_dims,
        requested_selectors=requested_selectors,
        dim_coord_values=dim_coord_values,
        allowed_dim_indices=allowed_dim_indices,
        allowed_chunk_ids_by_dim=allowed_chunk_ids_by_dim,
        feature_ds=feature_ds,
        feature_id_pa_type=feature_id_pa_type,
        feature_num_blocks=feature_num_blocks,
        feature_map_kwargs=feature_map_kwargs,
        chunk_kwargs=chunk_kwargs,
        group_keys=group_keys,
        group_key_pa_types=group_key_pa_types,
        aggs=aggs,
        plan_elapsed_s=plan_elapsed_s,
        scale_factor=scale_factor,
        add_offset=add_offset,
    )


def _log_zonal_stats_plan(
    *,
    plan: ZonalStatsPlan,
    store_uri: str,
    array_name: str | None,
    shuffle_num_partitions: int | None,
    feature_override_num_blocks: int | None,
    decode_coords: bool,
    all_touched: bool,
) -> None:
    logger.info(
        "zonal_stats planned store=%s array=%s dims=%s shape=%s chunk_sizes=%s "
        "non_spatial_dims=%s stats=%s shuffle_num_partitions=%s "
        "feature_override_num_blocks=%s feature_num_blocks=%s "
        "decode_coords=%s selectors=%s selected_index_counts=%s all_touched=%s plan_time_s=%.3f",
        store_uri,
        array_name or "<root-array>",
        plan.grid_spec.dims,
        plan.grid_spec.shape,
        plan.grid_spec.chunk_sizes,
        plan.non_spatial_dims,
        plan.resolved.requested,
        shuffle_num_partitions,
        feature_override_num_blocks,
        plan.feature_num_blocks,
        decode_coords,
        plan.requested_selectors,
        {dim: len(indices) for dim, indices in plan.allowed_dim_indices.items()},
        all_touched,
        plan.plan_elapsed_s,
    )


def _check_feature_crs(
    features: gpd.GeoDataFrame | str | Path | ray.data.Dataset,
    crs: pyproj.CRS,
) -> None:
    try:
        import geopandas  # noqa: F811
    except ImportError:
        return
    if not isinstance(features, geopandas.GeoDataFrame):
        return
    if features.crs is None:
        return
    feature_crs = pyproj.CRS(features.crs)
    if not feature_crs.equals(crs):
        raise ValueError(
            f"Feature CRS ({feature_crs.to_epsg() or feature_crs.to_wkt()}) does not match "
            f"the raster CRS ({crs.to_epsg() or crs.to_wkt()}). "
            "Reproject your features to match the raster CRS before calling zonal_stats."
        )


def _resolve_allowed_dim_indices(
    *,
    grid_spec: GridSpec,
    non_spatial_dims: list[str],
    dim_coord_values: dict[str, list],
    selectors: Mapping[str, object],
) -> dict[str, list[int]]:
    allowed: dict[str, list[int]] = {}
    for dim in non_spatial_dims:
        dim_size = grid_spec.shape[grid_spec.dim_index(dim)]
        selector = selectors.get(dim)
        if selector is None:
            allowed[dim] = list(range(dim_size))
            continue
        coord_values = dim_coord_values.get(dim, list(range(dim_size)))
        indices = _resolve_dim_selector_indices(
            coord_values=coord_values,
            dim_size=dim_size,
            selector=selector,
            dim_name=dim,
        )
        if not indices:
            raise ValueError(f"Selector for dimension {dim!r} matched no indices: {selector!r}")
        allowed[dim] = indices
    return allowed


def _resolve_allowed_chunk_ids_by_dim(
    *,
    grid_spec: GridSpec,
    allowed_dim_indices: Mapping[str, list[int]],
) -> dict[str, list[int]]:
    allowed_chunk_ids: dict[str, list[int]] = {}
    for dim, indices in allowed_dim_indices.items():
        axis = grid_spec.dim_index(dim)
        chunk_size = grid_spec.chunk_sizes[axis]
        allowed_chunk_ids[dim] = sorted({int(index) // int(chunk_size) for index in indices})
    return allowed_chunk_ids


def _resolve_dim_selector_indices(
    *,
    coord_values: Sequence[object],
    dim_size: int,
    selector: object,
    dim_name: str,
) -> list[int]:
    if isinstance(selector, int):
        index = selector if selector >= 0 else dim_size + selector
        if index < 0 or index >= dim_size:
            raise IndexError(f"Selector index out of bounds for dimension {dim_name!r}: {selector}")
        return [int(index)]

    if isinstance(selector, slice):
        if all(
            part is None or isinstance(part, int)
            for part in (selector.start, selector.stop, selector.step)
        ):
            return list(range(*selector.indices(dim_size)))
        return _resolve_coord_selection_with_xarray(
            coord_values=coord_values,
            selector=selector,
            dim_name=dim_name,
        )

    return _resolve_coord_selection_with_xarray(
        coord_values=coord_values,
        selector=selector,
        dim_name=dim_name,
    )


def _resolve_coord_selection_with_xarray(
    *,
    coord_values: Sequence[object],
    selector: object,
    dim_name: str,
) -> list[int]:
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "Label-based selectors require xarray. "
            "Install with the 'xarray' extra, for example "
            '`pip install "rayzon[xarray]"` or '
            '`uv pip install "rayzon[xarray]"`.'
        ) from exc

    positions = xr.DataArray(
        np.arange(len(coord_values), dtype=np.int64),
        dims=(dim_name,),
        coords={dim_name: list(coord_values)},
    )
    selected = positions.sel({dim_name: selector})
    values = np.asarray(selected.values).reshape(-1)
    return [int(value) for value in values.tolist()]


def _infer_group_key_pa_type(values: list[object], *, decode_coords: bool) -> pa.DataType:
    if not decode_coords:
        return pa.int64()
    sample = next((value for value in values if value is not None), None)
    if sample is None:
        return pa.int64()
    return pa.array([sample]).type
