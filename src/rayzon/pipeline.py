from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pyproj
import ray.data
from affine import Affine

from rayzon.chunk_processor import process_chunk_group
from rayzon.index import map_feature_to_chunk_rows
from rayzon.logging_utils import ensure_basic_logging, get_logger
from rayzon.rasterize_backend import RasterizeBackend
from rayzon.stats import DEFAULT_STAT_EXPRS, _finalize_partial_rows_in_feature_group
from rayzon.types import COL_CHUNK_KEY, COL_FEATURE_ID, COL_GEOMETRY
from rayzon.zarr_backend import ZarrBackend
from rayzon.zonal_plan import _log_zonal_stats_plan, build_zonal_stats_plan

if TYPE_CHECKING:
    import geopandas as gpd


logger = get_logger(__name__)


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
    plan = build_zonal_stats_plan(
        store_uri,
        features=features,
        transform=transform,
        crs=crs,
        x_dim=x_dim,
        y_dim=y_dim,
        stats=stats,
        array_name=array_name,
        all_touched=all_touched,
        nodata=nodata,
        decode_coords=decode_coords,
        storage_options=storage_options,
        selectors=selectors,
        feature_override_num_blocks=feature_override_num_blocks,
        zarr_backend=zarr_backend,
        rasterize_backend=rasterize_backend,
    )
    _log_zonal_stats_plan(
        plan=plan,
        store_uri=store_uri,
        array_name=array_name,
        shuffle_num_partitions=shuffle_num_partitions,
        feature_override_num_blocks=feature_override_num_blocks,
        decode_coords=decode_coords,
        all_touched=all_touched,
    )

    return (
        plan.feature_ds.map_batches(
            map_feature_to_chunk_rows,  # type: ignore[arg-type]
            fn_kwargs={
                "feature_id_col": COL_FEATURE_ID,
                "geometry_col": COL_GEOMETRY,
                **plan.feature_map_kwargs,
            },
            batch_format="pyarrow",
            num_cpus=0.1,
        )
        .groupby(COL_CHUNK_KEY, num_partitions=shuffle_num_partitions)
        .map_groups(
            process_chunk_group,  # type: ignore[arg-type]
            fn_kwargs=plan.chunk_kwargs,
            batch_format="pyarrow",
            num_cpus=1,
        )
        .groupby(COL_FEATURE_ID, num_partitions=shuffle_num_partitions)
        .map_groups(
            _finalize_partial_rows_in_feature_group,  # type: ignore[arg-type]
            fn_kwargs={
                "group_keys": plan.group_keys,
                "partial_columns": list(plan.partial_columns),
                "requested_stats": list(plan.resolved.requested),
            },
            batch_format="pyarrow",
            num_cpus=1,
        )
    )
