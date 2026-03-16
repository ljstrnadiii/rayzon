from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import ray.data

from rzzs.config import BenchmarkConfig, ZonalStatsConfig
from rzzs.grid import GridSpec
from rzzs.index import (
    ChunkFeatureIndex,
    _group_chunk_feature_rows,
    _map_feature_batch_to_chunk_rows,
    build_chunk_to_features_from_grouped_rows,
    build_feature_chunk_index_from_chunk_to_features,
    plan_chunk_jobs,
)
from rzzs.rasterize_backend import RasterizeBackend
from rzzs.stats import DEFAULT_STAT_EXPRS, resolve_stat_exprs
from rzzs.types import ChunkJob
from rzzs.zarr_backend import ZarrBackend, build_grid_spec_from_mosaic

if TYPE_CHECKING:
    import geopandas as gpd


@dataclass(frozen=True)
class PipelinePlan:
    grid_spec: GridSpec
    feature_index: ChunkFeatureIndex
    chunk_jobs: list[ChunkJob]


def geodataframe_to_geoarrow_table(frame: gpd.GeoDataFrame) -> pa.Table:
    return pa.table(frame.to_arrow())


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


def build_feature_chunk_index(
    feature_source: ray.data.Dataset,
    grid_spec: GridSpec,
    *,
    feature_id_col: str = "feature_id",
) -> ChunkFeatureIndex:
    results = (
        feature_source.map_batches(
            _map_feature_batch_to_chunk_rows,  # type: ignore[arg-type]
            fn_kwargs={"grid_spec": grid_spec, "feature_id_col": feature_id_col},
            batch_format="pyarrow",
        )
        .groupby("chunk_key")
        .map_groups(
            _group_chunk_feature_rows,  # type: ignore[arg-type]
            batch_format="pyarrow",
        )
    )

    chunk_to_features = build_chunk_to_features_from_grouped_rows(results.take_all())
    return build_feature_chunk_index_from_chunk_to_features(chunk_to_features)


def build_pipeline_plan(
    mosaic: str,
    features: gpd.GeoDataFrame | str | Path | ray.data.Dataset,
    *,
    config: ZonalStatsConfig,
    zarr_backend: ZarrBackend = ZarrBackend.ZARR_PYTHON,
) -> PipelinePlan:
    grid_spec = build_grid_spec_from_mosaic(
        mosaic,
        x_dim=config.x_dim,
        y_dim=config.y_dim,
        dst_crs=config.dst_crs,
        zarr_backend=zarr_backend,
    )
    feature_dataset = to_feature_dataset(features)
    feature_index = build_feature_chunk_index(feature_dataset, grid_spec)
    chunk_jobs = plan_chunk_jobs(feature_index)

    return PipelinePlan(
        grid_spec=grid_spec,
        feature_index=feature_index,
        chunk_jobs=chunk_jobs,
    )


def run_zonal_stats(
    mosaic: str,
    features: gpd.GeoDataFrame | str | Path | ray.data.Dataset,
    *,
    reduce_dims: Sequence[str],
    stats: Sequence[str] = DEFAULT_STAT_EXPRS,
    config: ZonalStatsConfig,
    output_uri: str,
    zarr_backend: ZarrBackend = ZarrBackend.ZARR_PYTHON,
    rasterize_backend: RasterizeBackend = RasterizeBackend.RASTERIO,
) -> str:
    """Run zonal statistics for a mosaic over vector features.

    Parameters
    ----------
    mosaic : str
        Store URI for the mosaic group. Default backend assumes a zarr group
        opened from an obstore-compatible store.
    features : geopandas.GeoDataFrame | str | pathlib.Path | ray.data.Dataset
        Feature source as a GeoDataFrame or a parquet path readable by Ray.
    reduce_dims : collections.abc.Sequence[str]
        Dimensions to reduce across.
    stats : collections.abc.Sequence[str]
        Optional stat expressions. When omitted, defaults to
        ('count', 'n_valid', 'mean', 'std').
    config : rzzs.config.ZonalStatsConfig
        Pipeline configuration.
    output_uri : str
        Output parquet sink URI.
    zarr_backend : rzzs.zarr_backend.ZarrBackend
        Zarr backend identifier. Defaults to zarr-python.
    rasterize_backend : rzzs.rasterize_backend.RasterizeBackend
        Rasterization backend identifier. Defaults to rasterio.
    """

    resolved_stats = resolve_stat_exprs(list(stats))

    plan = build_pipeline_plan(
        mosaic,
        features,
        config=config,
        zarr_backend=zarr_backend,
    )
    _ = plan, reduce_dims, resolved_stats, output_uri, rasterize_backend
    raise NotImplementedError(
        "Planning is implemented; chunk execution and final reduce/write are not implemented yet."
    )


def run_benchmark(
    mosaic: str,
    features: gpd.GeoDataFrame | str | Path | ray.data.Dataset,
    *,
    benchmark_config: BenchmarkConfig,
) -> dict[str, float | int]:
    """Run the benchmark harness for the configured workload.

    Parameters
    ----------
    mosaic : str
        Store URI for the mosaic group.
    features : geopandas.GeoDataFrame | str | pathlib.Path | ray.data.Dataset
        Feature source as a GeoDataFrame or a parquet path readable by Ray.
    benchmark_config : rzzs.config.BenchmarkConfig
        Benchmark execution configuration.
    """

    _ = mosaic, features, benchmark_config
    raise NotImplementedError("run_benchmark will be implemented in benchmark phase")
