from __future__ import annotations

from collections.abc import Sequence
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
)
from rzzs.stats import DEFAULT_STAT_EXPRS, resolve_stat_exprs

if TYPE_CHECKING:
    import geopandas as gpd


def geodataframe_to_geoarrow_table(frame: gpd.GeoDataFrame) -> pa.Table:
    return pa.table(frame.to_arrow())


def to_feature_dataset(
    feature_source: gpd.GeoDataFrame | str | Path | ray.data.Dataset,
) -> ray.data.Dataset:
    if isinstance(feature_source, ray.data.Dataset):
        return feature_source
    if isinstance(feature_source, (str, Path)):
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


def run_zonal_stats(
    mosaic: str,
    features: gpd.GeoDataFrame | str,
    *,
    reduce_dims: Sequence[str],
    stats: Sequence[str] = DEFAULT_STAT_EXPRS,
    config: ZonalStatsConfig,
    output_uri: str,
) -> str:
    """Run zonal statistics for a mosaic over vector features.

    Parameters
    ----------
    mosaic
        Store URI for the mosaic group. Default backend assumes a zarr group
        opened from an obstore-compatible store.
    features
        Feature source as a GeoDataFrame or a parquet path readable by Ray.
    reduce_dims
        Dimensions to reduce across.
    stats
        Optional stat expressions. When omitted, defaults to
        ('count', 'n_valid', 'mean', 'std').
    config
        Pipeline configuration.
    output_uri
        Output parquet sink URI.
    """

    requested_stats = tuple(stats)
    _resolved = resolve_stat_exprs(list(requested_stats))
    _ = mosaic, features, reduce_dims, _resolved, config, output_uri
    raise NotImplementedError("run_zonal_stats will be implemented in orchestration phase")


def run_benchmark(
    mosaic: str,
    features: gpd.GeoDataFrame | str,
    *,
    benchmark_config: BenchmarkConfig,
) -> dict[str, float | int]:
    """Run the benchmark harness for the configured workload."""

    _ = mosaic, features, benchmark_config
    raise NotImplementedError("run_benchmark will be implemented in benchmark phase")
