from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import zarr
from affine import Affine
from shapely.geometry import Point, box

from rzzs.config import ZonalStatsConfig
from rzzs.grid import GridSpec
from rzzs.index import plan_chunk_jobs
from rzzs.pipeline import (
    build_feature_chunk_index,
    build_pipeline_plan,
    geodataframe_to_geoarrow_table,
    to_feature_dataset,
)
from rzzs.zarr_backend import ZarrBackend


def _gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "feature_id": ["a", "b"],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def _grid() -> GridSpec:
    return GridSpec(
        dims=("y", "x"),
        shape=(100, 100),
        chunk_sizes=(10, 10),
        transform=Affine(1, 0, 0, 0, -1, 100),
        crs="EPSG:4326",
        x_dim="x",
        y_dim="y",
    )


def test_geodataframe_to_geoarrow_table_geometry_extension() -> None:
    table = geodataframe_to_geoarrow_table(_gdf())
    field = table.schema.field("geometry")
    metadata = field.metadata or {}

    assert metadata.get(b"ARROW:extension:name") == b"geoarrow.wkb"


def test_to_feature_dataset_from_geodataframe() -> None:
    dataset = to_feature_dataset(_gdf())
    assert dataset.count() == 2


def test_to_feature_dataset_geoarrow_extension_survives_in_ray_batch() -> None:
    dataset = to_feature_dataset(_gdf())
    first_batch = next(iter(dataset.iter_batches(batch_format="pyarrow", batch_size=10)))

    table = _as_table(first_batch)
    field = table.schema.field("geometry")
    metadata = field.metadata or {}
    assert metadata.get(b"ARROW:extension:name") == b"geoarrow.wkb"


def test_to_feature_dataset_from_geoparquet_path(tmp_path: Path) -> None:
    path = tmp_path / "features.parquet"
    _gdf().to_parquet(path, index=False)

    dataset = to_feature_dataset(path)
    assert dataset.count() == 2

    first_batch = next(iter(dataset.iter_batches(batch_format="pyarrow", batch_size=10)))
    table = _as_table(first_batch)
    field = table.schema.field("geometry")
    metadata = field.metadata or {}
    assert metadata.get(b"ARROW:extension:name") == b"geoarrow.wkb"


def test_build_feature_chunk_index_from_geodataframe() -> None:
    gdf = gpd.GeoDataFrame(
        {
            "feature_id": ["a"],
            "geometry": [box(1, 81, 19, 99)],
        },
        geometry="geometry",
    )
    dataset = to_feature_dataset(gdf)
    index = build_feature_chunk_index(dataset, _grid())
    assert "a" in index.feature_to_chunks
    assert len(index.feature_to_chunks["a"]) == 4


def test_build_feature_chunk_index_from_parquet_path(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "feature_id": ["a"],
            "xmin": [1.0],
            "ymin": [81.0],
            "xmax": [19.0],
            "ymax": [99.0],
        }
    )
    path = tmp_path / "polygons.parquet"
    df.to_parquet(path, index=False)

    dataset = to_feature_dataset(path)
    index = build_feature_chunk_index(dataset, _grid())
    assert "a" in index.feature_to_chunks
    assert len(index.feature_to_chunks["a"]) == 4


def test_plan_chunk_jobs_from_pipeline_index() -> None:
    index = build_feature_chunk_index(to_feature_dataset(_gdf()), _grid())
    jobs = plan_chunk_jobs(index)
    chunk_ids = [job["chunk_id"] for job in jobs]
    assert chunk_ids == sorted(chunk_ids)


def test_build_pipeline_plan_uses_zarr_backend(tmp_path: Path) -> None:
    mosaic_path = tmp_path / "mosaic.zarr"
    array = zarr.open_array(
        str(mosaic_path),
        mode="w",
        shape=(100, 100),
        chunks=(10, 10),
        dtype=np.float32,
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
    array.attrs["transform"] = [1.0, 0.0, 0.0, 0.0, -1.0, 100.0]
    array.attrs["crs"] = "EPSG:4326"

    features = gpd.GeoDataFrame(
        {
            "feature_id": ["a"],
            "geometry": [box(1, 81, 19, 99)],
        },
        geometry="geometry",
    )

    from rzzs.config import ZonalStatsConfig

    plan = build_pipeline_plan(
        str(mosaic_path),
        features,
        config=ZonalStatsConfig(x_dim="x", y_dim="y"),
        zarr_backend=ZarrBackend.ZARR_PYTHON,
    )

    assert plan.grid_spec.dims == ("y", "x")
    assert len(plan.chunk_jobs) == 4


def test_build_pipeline_plan_rejects_unknown_backends(tmp_path: Path) -> None:
    mosaic_path = tmp_path / "mosaic.zarr"
    zarr.open_array(
        str(mosaic_path),
        mode="w",
        shape=(10, 10),
        chunks=(5, 5),
        dtype=np.float32,
        zarr_format=3,
        dimension_names=("y", "x"),
    )

    features = gpd.GeoDataFrame(
        {
            "feature_id": ["a"],
            "geometry": [box(1, 1, 2, 2)],
        },
        geometry="geometry",
    )

    with pytest.raises(ValueError, match="Unsupported zarr backend"):
        build_pipeline_plan(
            str(mosaic_path),
            features,
            config=ZonalStatsConfig(x_dim="x", y_dim="y"),
            zarr_backend=cast(ZarrBackend, "unknown"),
        )


def _as_table(batch: object) -> pa.Table:
    if isinstance(batch, pa.Table):
        return batch
    if isinstance(batch, pa.RecordBatch):
        return pa.Table.from_batches([batch])
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")
