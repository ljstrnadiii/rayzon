from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, cast

import numpy as np
import pyarrow as pa
import pyproj
import ray.data
from affine import Affine

from rayzon.arrow import geodataframe_to_geoarrow_table
from rayzon.chunk_processor import process_chunk_group
from rayzon.grid import GridSpec
from rayzon.index import map_feature_to_chunk_rows
from rayzon.logging_utils import ensure_basic_logging, get_logger
from rayzon.rasterize_backend import RasterizeBackend
from rayzon.stats import (
    DEFAULT_STAT_EXPRS,
    build_aggregations,
    required_partial_columns,
    resolve_stat_exprs,
)
from rayzon.types import COL_CHUNK_KEY, COL_FEATURE_ID, COL_GEOMETRY
from rayzon.zarr_backend import ZarrBackend, build_grid_spec, resolve_dim_coords

if TYPE_CHECKING:
    import geopandas as gpd


logger = get_logger(__name__)


def to_feature_dataset(
    feature_source: gpd.GeoDataFrame | str | Path | ray.data.Dataset,
    *,
    override_num_blocks: int | None = None,
) -> ray.data.Dataset:
    if isinstance(feature_source, ray.data.Dataset):
        dataset = _maybe_override_num_blocks(feature_source, override_num_blocks)
        return _ensure_feature_id_column(dataset)
    if isinstance(feature_source, str | Path):
        dataset = ray.data.read_parquet(
            str(feature_source),
            include_paths=True,
            override_num_blocks=override_num_blocks,
        )
        return _ensure_feature_id_column(dataset)

    try:
        import geopandas as geopandas
    except ImportError as exc:
        raise ImportError(
            "GeoDataFrame support requires geopandas. "
            "Install with the 'geopandas' extra, for example "
            '`pip install "rayzon[geopandas]"` or '
            '`uv pip install "rayzon[geopandas]"`.'
        ) from exc

    if isinstance(feature_source, geopandas.GeoDataFrame):
        dataset = ray.data.from_arrow(geodataframe_to_geoarrow_table(feature_source))
        dataset = _maybe_override_num_blocks(dataset, override_num_blocks)
        return _ensure_feature_id_column(dataset)

    raise TypeError("feature_source must be a Ray Dataset, GeoDataFrame, or parquet path")


def zonal_stats(
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
    shuffle_num_partitions: int | None = None,
    zarr_backend: ZarrBackend = ZarrBackend.ZARR_PYTHON,
    rasterize_backend: RasterizeBackend = RasterizeBackend.RASTERIO,
) -> ray.data.Dataset:
    """Compute zonal statistics over a zarr array for a set of geometries.

    Parameters
    ----------
    store_uri : str
        URI of the zarr store.  For a standalone zarr array, point directly at the array.  For a
        zarr group (e.g. written by xarray), point at the group root and set *array_name*.
    features : GeoDataFrame | str | Path | ray.data.Dataset
        Geometries to compute statistics for. Must be in the same CRS as *crs*.
    transform : Affine | None
        The Affine transform of the raster data. If omitted, infer from array or root-group attrs.
    crs : pyproj.CRS | None
        Coordinate reference system of the raster data. If omitted, infer from attrs.
    x_dim, y_dim : str
        Names of the spatial dimensions in the zarr array.
    stats : Sequence[str]
        Stat expressions to compute, e.g. ``("count", "mean", "std")``.
    array_name : str | None
        Name of the array within a zarr group.  ``None`` when *store_uri* already points at a
        standalone array.
    all_touched : bool
        If True, all pixels touched by a geometry are included.
    nodata : float | int | None
        Pixel value to treat as missing (replaced with NaN before stats).
    decode_coords : bool
        If True, use xarray to decode coordinate values for non-spatial dimensions.  Requires the
        ``xarray`` extra. Otherwise, values in the final dataset will be the corresponding integer
        indices along each non-spatial dimension.
    storage_options : Mapping[str, object] | None
        Backend-specific storage options for opening the zarr store, e.g. ``{"anon": True}`` for
        public S3 buckets.
    selectors : Mapping[str, object] | None
        Optional non-spatial selectors used to subset the array before planning chunk work. Supports
        integer positional indices and xarray-style label-based selectors such as exact coordinate
        matches, datetime strings like ``{"time": "2023"}``, and coordinate slices.
    feature_override_num_blocks : int | None
        Override the initial Ray Dataset block count for the feature source. For parquet-path
        inputs this is passed to ``ray.data.read_parquet(..., override_num_blocks=...)``.
        For GeoDataFrame and Ray Dataset inputs, the loaded dataset is repartitioned to this block
        count. Use this to control upstream feature-task parallelism independently of downstream
        groupby shuffle partitioning.
    shuffle_num_partitions : int | None
        Number of hash-shuffle partitions to use for the internal Ray Dataset groupby stages.
        Set this explicitly for local runs to avoid oversharding, e.g. ``8``.
    zarr_backend : ZarrBackend
        Backend for reading zarr data. Defaults to ZARR_PYTHON.
    rasterize_backend : RasterizeBackend
        Backend for rasterizing geometries. Defaults to RASTERIO.

    Returns
    -------
    ray.data.Dataset
        One row per (feature, non-spatial dim combination) with the requested
        stat columns.
    """
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
    aggs = build_aggregations(resolved)
    logger.info(
        "zonal_stats planned store=%s array=%s dims=%s shape=%s chunk_sizes=%s "
        "non_spatial_dims=%s stats=%s shuffle_num_partitions=%s "
        "feature_override_num_blocks=%s feature_num_blocks=%s "
        "decode_coords=%s selectors=%s selected_index_counts=%s all_touched=%s plan_time_s=%.3f",
        store_uri,
        array_name or "<root-array>",
        grid_spec.dims,
        grid_spec.shape,
        grid_spec.chunk_sizes,
        non_spatial_dims,
        resolved.requested,
        shuffle_num_partitions,
        feature_override_num_blocks,
        feature_num_blocks,
        decode_coords,
        requested_selectors,
        {dim: len(indices) for dim, indices in allowed_dim_indices.items()},
        all_touched,
        plan_elapsed_s,
    )

    return (
        feature_ds.map_batches(
            map_feature_to_chunk_rows,  # type: ignore[arg-type]
            fn_kwargs={
                "feature_id_col": COL_FEATURE_ID,
                "geometry_col": COL_GEOMETRY,
                **feature_map_kwargs,
            },
            batch_format="pyarrow",
            num_cpus=0.1,
        )
        .groupby(COL_CHUNK_KEY, num_partitions=shuffle_num_partitions)
        .map_groups(
            process_chunk_group,  # type: ignore[arg-type]
            fn_kwargs=chunk_kwargs,
            batch_format="pyarrow",
            num_cpus=1,
        )
        .groupby(group_keys, num_partitions=shuffle_num_partitions)
        .aggregate(*aggs)
    )


def _detect_feature_id_type(ds: ray.data.Dataset) -> pa.DataType:
    schema = ds.schema()
    if schema is None:
        return pa.string()

    arrow_schema = getattr(schema, "base_schema", None)
    if arrow_schema is not None and hasattr(arrow_schema, "field"):
        try:
            return arrow_schema.field(COL_FEATURE_ID).type
        except (KeyError, AttributeError):
            pass

    if hasattr(schema, "field"):
        try:
            return schema.field(COL_FEATURE_ID).type
        except (KeyError, AttributeError):
            pass
    return pa.string()


def _maybe_override_num_blocks(
    ds: ray.data.Dataset,
    override_num_blocks: int | None,
) -> ray.data.Dataset:
    if override_num_blocks is None:
        return ds
    if override_num_blocks <= 0:
        raise ValueError("override_num_blocks must be a positive integer when provided.")
    return ds.repartition(override_num_blocks)


def _detect_num_blocks(ds: ray.data.Dataset) -> int | None:
    try:
        return int(ds.num_blocks())
    except NotImplementedError:
        return None


def _ensure_feature_id_column(ds: ray.data.Dataset) -> ray.data.Dataset:
    schema = ds.schema()
    if schema is None or not hasattr(schema, "names"):
        return ds

    column_names = list(schema.names)
    if COL_FEATURE_ID in column_names:
        return ds

    logger.info(
        "feature_id missing; synthesizing integer %s by zipping with row range",
        COL_FEATURE_ID,
    )
    materialized = ds.materialize()
    row_index_ds = ray.data.range(
        materialized.count(),
        override_num_blocks=materialized.num_blocks(),
    ).rename_columns({"id": "__rayzon_feature_index"})
    result = materialized.zip(row_index_ds).rename_columns(
        {"__rayzon_feature_index": COL_FEATURE_ID}
    )
    return cast("ray.data.Dataset", result)


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
