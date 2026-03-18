from __future__ import annotations

import math
from typing import cast

import numpy as np
import pyarrow as pa
import zarr
from affine import Affine
from shapely import from_wkb
from shapely.geometry.base import BaseGeometry

from rayzon.arrow import as_table
from rayzon.grid import GridSpec, chunk_id_to_slices, chunk_world_bounds, reconstruct_grid_spec
from rayzon.index import _chunk_key_to_chunk_id, _normalize_feature_id
from rayzon.rasterize_backend import RasterizeBackend, rasterize_geometry_window
from rayzon.stats import (
    ALL_PARTIAL_COLUMNS,
    accumulate_partial_state,
    partial_column_arrow_type,
    partial_fields_for_columns,
)
from rayzon.types import (
    COL_CHUNK_KEY,
    COL_FEATURE_ID,
    COL_GEOMETRY,
)
from rayzon.zarr_backend import ZarrBackend, open_zarr_array


def process_chunk(
    chunk_id: tuple[int, ...],
    feature_ids: list[int | str],
    array: zarr.Array,
    geometry_store: dict[int | str, BaseGeometry],
    grid_spec: GridSpec,
    *,
    all_touched: bool = False,
    nodata: float | int | None = None,
    scale_factor: float | None = None,
    add_offset: float | None = None,
    partial_columns: tuple[str, ...] = ALL_PARTIAL_COLUMNS,
    rasterize_backend: RasterizeBackend = RasterizeBackend.RASTERIO,
) -> list[dict[str, object]]:
    partial_fields = partial_fields_for_columns(partial_columns)
    chunk_slices = chunk_id_to_slices(chunk_id, grid_spec)
    chunk_data = np.asarray(array[chunk_slices])

    x0, y0, x1, y1 = _chunk_xy_bounds(chunk_id, grid_spec)
    chunk_bounds = chunk_world_bounds(chunk_id, grid_spec)

    rows: list[dict[str, object]] = []
    for feature_id in feature_ids:
        geometry = geometry_store.get(feature_id)
        if geometry is None:
            continue

        if not _bbox_intersects(geometry.bounds, chunk_bounds):
            continue

        local_window = _geometry_local_window(geometry.bounds, grid_spec, x0, y0, x1, y1)
        if local_window is None:
            continue

        lx0, ly0, lx1, ly1 = local_window
        out_shape = (ly1 - ly0, lx1 - lx0)
        if out_shape[0] <= 0 or out_shape[1] <= 0:
            continue

        window_transform = Affine(*grid_spec.transform) * Affine.translation(x0 + lx0, y0 + ly0)
        effective_all_touched = _effective_all_touched(all_touched, geometry.geom_type)
        mask = rasterize_geometry_window(
            geometry,
            out_shape=out_shape,
            transform=window_transform,
            all_touched=effective_all_touched,
            rasterize_backend=rasterize_backend,
        )
        if not np.any(mask):
            continue

        windowed = chunk_data[
            _windowed_chunk_slices(chunk_data.ndim, grid_spec, lx0, ly0, lx1, ly1)
        ]

        for dim_values, plane in _iter_spatial_planes(windowed, grid_spec, chunk_slices):
            selected = np.asarray(plane)[mask]
            if selected.size == 0:
                continue

            if nodata is not None:
                selected = np.where(selected == nodata, np.nan, selected)

            if (scale_factor is not None and scale_factor != 1.0) or (
                add_offset is not None and add_offset != 0.0
            ):
                selected = selected * (scale_factor or 1.0) + (add_offset or 0.0)

            partial_state = accumulate_partial_state(selected, required_fields=partial_fields)
            if partial_state is None:
                continue
            row: dict[str, object] = {
                "feature_id": feature_id,
                "dim_values": dim_values,
            }
            for column in partial_columns:
                row[column] = partial_state[column]
            rows.append(row)

    return rows


def _iter_spatial_planes(
    windowed: np.ndarray,
    grid_spec: GridSpec,
    chunk_slices: tuple[slice, ...],
) -> list[tuple[tuple[int, ...], np.ndarray]]:
    non_spatial_axes = tuple(
        idx for idx in range(windowed.ndim) if idx not in (grid_spec.y_index, grid_spec.x_index)
    )
    if not non_spatial_axes:
        return [((), windowed)]

    non_spatial_shape = tuple(windowed.shape[idx] for idx in non_spatial_axes)
    out: list[tuple[tuple[int, ...], np.ndarray]] = []
    for local_idx in np.ndindex(non_spatial_shape):
        slicer: list[int | slice] = [slice(None)] * windowed.ndim
        dim_values: list[int] = []
        for axis, axis_local_idx in zip(non_spatial_axes, local_idx, strict=True):
            slicer[axis] = axis_local_idx
            axis_start = chunk_slices[axis].start or 0
            dim_values.append(axis_start + axis_local_idx)
        out.append((tuple(dim_values), np.asarray(windowed[tuple(slicer)])))
    return out


def _chunk_xy_bounds(chunk_id: tuple[int, ...], grid_spec: GridSpec) -> tuple[int, int, int, int]:
    chunk_slices = chunk_id_to_slices(chunk_id, grid_spec)
    ys = chunk_slices[grid_spec.y_index]
    xs = chunk_slices[grid_spec.x_index]
    return (xs.start or 0, ys.start or 0, xs.stop or 0, ys.stop or 0)


def _geometry_local_window(
    bbox: tuple[float, float, float, float],
    grid_spec: GridSpec,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[int, int, int, int] | None:
    inv = ~Affine(*grid_spec.transform)
    minx, miny, maxx, maxy = bbox
    px0, py0 = inv * (minx, maxy)
    px1, py1 = inv * (maxx, miny)

    gx0 = max(x0, math.floor(min(px0, px1)))
    gx1 = min(x1, math.ceil(max(px0, px1)))
    gy0 = max(y0, math.floor(min(py0, py1)))
    gy1 = min(y1, math.ceil(max(py0, py1)))
    if gx0 >= gx1 or gy0 >= gy1:
        return None

    return (gx0 - x0, gy0 - y0, gx1 - x0, gy1 - y0)


def _windowed_chunk_slices(
    ndim: int,
    grid_spec: GridSpec,
    lx0: int,
    ly0: int,
    lx1: int,
    ly1: int,
) -> tuple[slice, ...]:
    slices: list[slice] = [slice(None)] * ndim
    slices[grid_spec.x_index] = slice(lx0, lx1)
    slices[grid_spec.y_index] = slice(ly0, ly1)
    return tuple(slices)


def _bbox_intersects(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return not (
        left[2] <= right[0] or left[0] >= right[2] or left[3] <= right[1] or left[1] >= right[3]
    )


def _effective_all_touched(base_all_touched: bool, geom_type: str) -> bool:
    if geom_type in {"LineString", "MultiLineString"}:
        return True
    return base_all_touched


def process_chunk_group(
    batch: pa.Table,
    *,
    store_uri: str,
    array_name: str,
    all_touched: bool,
    nodata: float | int | None,
    zarr_backend_value: str,
    rasterize_backend_value: str,
    partial_columns: list[str],
    feature_id_pa_type: str,
    scale_factor: float | None,
    add_offset: float | None,
    dim_coord_values: dict[str, list],
    dims: list[str],
    shape: list[int],
    chunk_sizes: list[int],
    transform_coeffs: list[float],
    crs: str,
    x_dim: str,
    y_dim: str,
) -> pa.Table:
    partial_columns_tuple = tuple(partial_columns)
    grid_spec = reconstruct_grid_spec(
        dims=dims,
        shape=shape,
        chunk_sizes=chunk_sizes,
        transform_coeffs=transform_coeffs,
        crs=crs,
        x_dim=x_dim,
        y_dim=y_dim,
    )
    zarr_backend = ZarrBackend(zarr_backend_value)
    rasterize_backend = RasterizeBackend(rasterize_backend_value)
    non_spatial_dims = [d for d in grid_spec.dims if d not in (y_dim, x_dim)]
    fid_pa_type = _resolve_fid_type(feature_id_pa_type)

    table = as_table(batch)
    if table.num_rows == 0:
        return _empty_batch(
            non_spatial_dims,
            fid_pa_type=fid_pa_type,
            partial_columns=partial_columns_tuple,
        )

    chunk_key = table.column(COL_CHUNK_KEY)[0].as_py()
    chunk_id = _chunk_key_to_chunk_id(chunk_key)

    feature_ids_raw = table.column(COL_FEATURE_ID).to_pylist()
    geom_wkbs = table.column(COL_GEOMETRY).to_pylist()

    geometry_store: dict[int | str, BaseGeometry] = {}
    for fid, gwkb in zip(feature_ids_raw, geom_wkbs, strict=True):
        fid_norm = _normalize_feature_id(fid)
        if fid_norm not in geometry_store:
            geometry_store[fid_norm] = from_wkb(bytes(gwkb))

    array = open_zarr_array(
        store_uri,
        array_name=array_name or None,
        zarr_backend=zarr_backend,
    )
    rows = process_chunk(
        chunk_id=chunk_id,
        feature_ids=list(geometry_store.keys()),
        array=array,
        geometry_store=geometry_store,
        grid_spec=grid_spec,
        all_touched=all_touched,
        nodata=nodata,
        scale_factor=scale_factor,
        add_offset=add_offset,
        partial_columns=partial_columns_tuple,
        rasterize_backend=rasterize_backend,
    )
    return _rows_to_batch(
        rows,
        non_spatial_dims,
        fid_pa_type=fid_pa_type,
        partial_columns=partial_columns_tuple,
        dim_coord_values=dim_coord_values,
    )


def _resolve_fid_type(type_str: str) -> pa.DataType:
    factory = getattr(pa, type_str, None)
    if factory is not None and callable(factory):
        return factory()
    return pa.string()


def _empty_batch(
    non_spatial_dims: list[str],
    *,
    fid_pa_type: pa.DataType,
    partial_columns: tuple[str, ...],
    dim_coord_values: dict[str, list] | None = None,
) -> pa.Table:
    columns: dict[str, pa.Array] = {COL_FEATURE_ID: pa.array([], type=fid_pa_type)}
    for column in partial_columns:
        columns[column] = pa.array([], type=partial_column_arrow_type(column))
    for dim in non_spatial_dims:
        if dim_coord_values and dim in dim_coord_values and dim_coord_values[dim]:
            sample = dim_coord_values[dim][0]
            columns[dim] = pa.array([], type=pa.array([sample]).type)
        else:
            columns[dim] = pa.array([], type=pa.int64())
    return pa.table(columns)


def _rows_to_batch(
    rows: list[dict[str, object]],
    non_spatial_dims: list[str],
    *,
    fid_pa_type: pa.DataType,
    partial_columns: tuple[str, ...],
    dim_coord_values: dict[str, list] | None = None,
) -> pa.Table:
    if not rows:
        return _empty_batch(
            non_spatial_dims,
            fid_pa_type=fid_pa_type,
            partial_columns=partial_columns,
            dim_coord_values=dim_coord_values,
        )
    columns: dict[str, pa.Array] = {
        COL_FEATURE_ID: pa.array([r["feature_id"] for r in rows], type=fid_pa_type)
    }
    for column in partial_columns:
        columns[column] = pa.array(
            [r.get(column) for r in rows],
            type=partial_column_arrow_type(column),
        )
    for dim_idx, dim in enumerate(non_spatial_dims):
        raw_indices = [cast(tuple[int, ...], r["dim_values"])[dim_idx] for r in rows]
        if dim_coord_values and dim in dim_coord_values:
            coord_list = dim_coord_values[dim]
            columns[dim] = pa.array([coord_list[int(i)] for i in raw_indices])
        else:
            columns[dim] = pa.array(raw_indices, type=pa.int64())
    return pa.table(columns)
