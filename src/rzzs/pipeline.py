from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyproj
import ray.data
from affine import Affine

from rzzs.arrow import geodataframe_to_geoarrow_table
from rzzs.chunk_processor import process_chunk_group
from rzzs.index import map_feature_to_chunk_rows
from rzzs.rasterize_backend import RasterizeBackend
from rzzs.stats import DEFAULT_STAT_EXPRS, build_aggregations, resolve_stat_exprs
from rzzs.types import COL_CHUNK_KEY, COL_FEATURE_ID, COL_GEOMETRY
from rzzs.zarr_backend import ZarrBackend, build_grid_spec, resolve_dim_coords

if TYPE_CHECKING:
    import geopandas as gpd


def to_feature_dataset(
    feature_source: gpd.GeoDataFrame | str | Path | ray.data.Dataset,
) -> ray.data.Dataset:
    if isinstance(feature_source, ray.data.Dataset):
        return feature_source
    if isinstance(feature_source, str | Path):
        return ray.data.read_parquet(str(feature_source))

    try:
        import geopandas as geopandas
    except ImportError as exc:
        raise ImportError(
            "GeoDataFrame support requires geopandas. "
            "Install with the 'geopandas' extra, for example "
            '`pip install "rzzs[geopandas]"` or '
            '`uv pip install "rzzs[geopandas]"`.'
        ) from exc

    if isinstance(feature_source, geopandas.GeoDataFrame):
        return ray.data.from_arrow(geodataframe_to_geoarrow_table(feature_source))

    raise TypeError("feature_source must be a Ray Dataset, GeoDataFrame, or parquet path")


def zonal_stats(
    store_uri: str,
    features: gpd.GeoDataFrame | str | Path | ray.data.Dataset,
    *,
    transform: Affine,
    crs: pyproj.CRS,
    x_dim: str = "x",
    y_dim: str = "y",
    stats: Sequence[str] = DEFAULT_STAT_EXPRS,
    array_name: str | None = None,
    all_touched: bool = False,
    nodata: float | int | None = None,
    decode_coords: bool = False,
    zarr_backend: ZarrBackend = ZarrBackend.ZARR_PYTHON,
    rasterize_backend: RasterizeBackend = RasterizeBackend.RASTERIO,
) -> ray.data.Dataset:
    """Compute zonal statistics over a zarr array for a set of geometries.

    Parameters
    ----------
    store_uri : str
        URI of the zarr store.  For a standalone zarr array, point directly
        at the array.  For a zarr group (e.g. written by xarray), point at
        the group root and set *array_name*.
    features : GeoDataFrame | str | Path | ray.data.Dataset
        Geometries to compute statistics for. Must be in the same CRS as
        *crs*.
    transform : Affine
        Affine transform mapping pixel coordinates to the CRS.
    crs : pyproj.CRS
        Coordinate reference system of the raster data.
    x_dim, y_dim : str
        Names of the spatial dimensions in the zarr array.
    stats : Sequence[str]
        Stat expressions to compute, e.g. ``("count", "mean", "std")``.
    array_name : str | None
        Name of the array within a zarr group.  ``None`` when *store_uri*
        already points at a standalone array.
    all_touched : bool
        If True, all pixels touched by a geometry are included.
    nodata : float | int | None
        Pixel value to treat as missing (replaced with NaN before stats).
    decode_coords : bool
        If True, use xarray to decode coordinate values for non-spatial
        dimensions.  Requires the ``xarray`` extra.
    zarr_backend : ZarrBackend
        Backend for reading zarr data.
    rasterize_backend : RasterizeBackend
        Backend for rasterizing geometries.

    Returns
    -------
    ray.data.Dataset
        One row per (feature, non-spatial dim combination) with the requested
        stat columns.
    """
    resolved = resolve_stat_exprs(list(stats))
    transform_tuple = (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
    crs_wkt = crs.to_wkt()

    grid_spec, pixel_dtype, scale_factor, add_offset = build_grid_spec(
        store_uri,
        x_dim=x_dim,
        y_dim=y_dim,
        transform=transform_tuple,
        crs=crs_wkt,
        array_name=array_name,
        zarr_backend=zarr_backend,
    )
    non_spatial_dims = [dim for dim in grid_spec.dims if dim not in (y_dim, x_dim)]

    dim_coord_values: dict[str, list] = {}
    if decode_coords and non_spatial_dims:
        dim_coord_values = resolve_dim_coords(
            store_uri,
            dims=grid_spec.dims,
            x_dim=x_dim,
            y_dim=y_dim,
            array_name=array_name,
        )

    _check_feature_crs(features, crs)
    feature_ds = to_feature_dataset(features)
    feature_id_pa_type = _detect_feature_id_type(feature_ds)

    grid_kwargs: dict[str, object] = {
        "dims": list(grid_spec.dims),
        "shape": list(grid_spec.shape),
        "chunk_sizes": list(grid_spec.chunk_sizes),
        "transform_coeffs": list(grid_spec.transform),
        "crs": grid_spec.crs,
        "x_dim": x_dim,
        "y_dim": y_dim,
    }
    chunk_kwargs = {
        "store_uri": store_uri,
        "array_name": array_name or "",
        "all_touched": all_touched,
        "nodata": nodata,
        "zarr_backend_value": zarr_backend.value,
        "rasterize_backend_value": rasterize_backend.value,
        "pixel_numpy_dtype": str(pixel_dtype),
        "feature_id_pa_type": str(feature_id_pa_type),
        "scale_factor": scale_factor,
        "add_offset": add_offset,
        "dim_coord_values": dim_coord_values,
        **grid_kwargs,
    }

    group_keys = [COL_FEATURE_ID] + non_spatial_dims
    aggs = build_aggregations(resolved)

    return (
        feature_ds.map_batches(
            map_feature_to_chunk_rows,  # type: ignore[arg-type]
            fn_kwargs={
                "feature_id_col": COL_FEATURE_ID,
                "geometry_col": COL_GEOMETRY,
                **grid_kwargs,
            },
            batch_format="pyarrow",
        )
        .groupby(COL_CHUNK_KEY)
        .map_groups(
            process_chunk_group,  # type: ignore[arg-type]
            fn_kwargs=chunk_kwargs,
            batch_format="pyarrow",
        )
        .groupby(group_keys)
        .aggregate(*aggs)
    )


def _detect_feature_id_type(ds: ray.data.Dataset) -> pa.DataType:
    schema = ds.schema()
    if schema is not None and hasattr(schema, "field"):
        try:
            return schema.field(COL_FEATURE_ID).type
        except (KeyError, AttributeError):
            pass
    return pa.string()


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
