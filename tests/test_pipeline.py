from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import zarr
from affine import Affine
from shapely.geometry import Point, box

from rzzs.arrow import as_table, geodataframe_to_geoarrow_table
from rzzs.pipeline import (
    to_feature_dataset,
    zonal_stats,
)
from rzzs.types import COL_FEATURE_ID


def _gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            COL_FEATURE_ID: ["a", "b"],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
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

    table = as_table(first_batch)
    field = table.schema.field("geometry")
    metadata = field.metadata or {}
    assert metadata.get(b"ARROW:extension:name") == b"geoarrow.wkb"


def test_to_feature_dataset_from_geoparquet_path(tmp_path: Path) -> None:
    path = tmp_path / "features.parquet"
    _gdf().to_parquet(path, index=False)

    dataset = to_feature_dataset(path)
    assert dataset.count() == 2

    first_batch = next(iter(dataset.iter_batches(batch_format="pyarrow", batch_size=10)))
    table = as_table(first_batch)
    field = table.schema.field("geometry")
    metadata = field.metadata or {}
    assert metadata.get(b"ARROW:extension:name") == b"geoarrow.wkb"


def test_zonal_stats_smoke(tmp_path: Path) -> None:
    mosaic_path = tmp_path / "mosaic.zarr"
    array = zarr.open_array(
        str(mosaic_path),
        mode="w",
        zarr_format=3,
        shape=(4, 4),
        chunks=(4, 4),
        dtype=np.float32,
        dimension_names=("y", "x"),
    )
    array[:] = np.arange(16, dtype=np.float32).reshape(4, 4)
    array.attrs["transform"] = [1.0, 0.0, 0.0, 0.0, -1.0, 4.0]
    array.attrs["crs"] = "EPSG:4326"

    features = gpd.GeoDataFrame(
        {
            COL_FEATURE_ID: ["a"],
            "geometry": [box(1.0, 1.0, 3.0, 3.0)],
        },
        geometry="geometry",
    )

    output_path = tmp_path / "stats.parquet"
    result_ds = zonal_stats(
        str(mosaic_path),
        features,
        transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
        crs=pyproj.CRS("EPSG:4326"),
        stats=("count", "sum", "mean"),
    )

    result_ds.write_parquet(str(output_path))
    assert output_path.exists()
    out = pd.read_parquet(output_path)
    assert set(out.columns) >= {COL_FEATURE_ID, "count", "sum", "mean"}
    assert "n_valid" not in out.columns  # not explicitly requested
    assert out.iloc[0][COL_FEATURE_ID] == "a"


def test_zonal_stats_4d_multi_geom_multi_chunk(tmp_path: Path) -> None:
    mosaic_path = tmp_path / "mosaic4d.zarr"
    nt, nb, ny, nx = 3, 2, 8, 8
    array = zarr.open_array(
        str(mosaic_path),
        mode="w",
        zarr_format=3,
        shape=(nt, nb, ny, nx),
        chunks=(1, 1, 4, 4),
        dtype=np.float32,
        dimension_names=("time", "band", "y", "x"),
    )
    # Fill with constant 1.0 so sum == n_valid for easy verification
    array[:] = np.ones((nt, nb, ny, nx), dtype=np.float32)
    # transform: 1 pixel = 1 degree, top-left at (0, 8)
    array.attrs["transform"] = [1.0, 0.0, 0.0, 0.0, -1.0, 8.0]
    array.attrs["crs"] = "EPSG:4326"

    # Feature a: box(1,5,3,7) → 2x2 pixels, sits inside one spatial chunk
    # Feature b: box(0,0,8,8) → covers all 8x8 pixels, spans 4 spatial chunks
    features = gpd.GeoDataFrame(
        {
            COL_FEATURE_ID: ["a", "b"],
            "geometry": [box(1, 5, 3, 7), box(0, 0, 8, 8)],
        },
        geometry="geometry",
    )

    result_ds = zonal_stats(
        str(mosaic_path),
        features,
        transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 8.0),
        crs=pyproj.CRS("EPSG:4326"),
        stats=("count", "sum", "mean", "min", "max"),
    )

    output_path = tmp_path / "stats4d.parquet"
    result_ds.write_parquet(str(output_path))
    out = pd.read_parquet(output_path)

    assert set(out.columns) >= {COL_FEATURE_ID, "count", "sum", "mean", "min", "max"}
    # Should have one row per (feature, time, band) = 2 * 3 * 2 = 12
    assert len(out) == 12

    a_rows = out[out[COL_FEATURE_ID] == "a"]
    b_rows = out[out[COL_FEATURE_ID] == "b"]
    assert len(a_rows) == nt * nb  # 6
    assert len(b_rows) == nt * nb  # 6

    # Feature a covers 2x2=4 pixels, all 1.0
    for _, row in a_rows.iterrows():
        assert row["count"] == 4
        assert row["sum"] == 4.0
        assert row["mean"] == 1.0
        assert row["min"] == 1.0
        assert row["max"] == 1.0

    # Feature b covers 8x8=64 pixels, all 1.0
    for _, row in b_rows.iterrows():
        assert row["count"] == 64
        assert row["sum"] == 64.0
        assert row["mean"] == 1.0
        assert row["min"] == 1.0
        assert row["max"] == 1.0


def test_zonal_stats_only_requested_columns(tmp_path: Path) -> None:
    mosaic_path = tmp_path / "mosaic_cols.zarr"
    array = zarr.open_array(
        str(mosaic_path),
        mode="w",
        zarr_format=3,
        shape=(4, 4),
        chunks=(4, 4),
        dtype=np.float32,
        dimension_names=("y", "x"),
    )
    array[:] = np.ones((4, 4), dtype=np.float32)
    array.attrs["transform"] = [1.0, 0.0, 0.0, 0.0, -1.0, 4.0]
    array.attrs["crs"] = "EPSG:4326"

    features = gpd.GeoDataFrame(
        {COL_FEATURE_ID: ["a"], "geometry": [box(0, 0, 4, 4)]},
        geometry="geometry",
    )

    result_ds = zonal_stats(
        str(mosaic_path),
        features,
        transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
        crs=pyproj.CRS("EPSG:4326"),
        stats=("count",),
    )

    output_path = tmp_path / "stats_cols.parquet"
    result_ds.write_parquet(str(output_path))
    out = pd.read_parquet(output_path)

    assert "count" in out.columns
    assert out.iloc[0]["count"] == 16
    # Should NOT have stats that weren't requested
    for col in ("sum", "sum_sq", "mean", "variance", "std", "min", "max"):
        assert col not in out.columns, f"unexpected column: {col}"


def test_zonal_stats_decode_coords(tmp_path: Path) -> None:
    import xarray as xr

    mosaic_path = tmp_path / "mosaic_ts.zarr"
    times = pd.date_range("2020-01-01", periods=3, freq="D")
    ds = xr.Dataset(
        {"data": (("time", "band", "y", "x"), np.ones((3, 2, 4, 4), dtype=np.float32))},
        coords={
            "time": times,
            "band": ["b1", "b2"],
            "y": np.arange(4.0, 0.0, -1.0),
            "x": np.arange(0.0, 4.0),
        },
    )
    ds["data"].attrs["transform"] = [1.0, 0.0, 0.0, 0.0, -1.0, 4.0]
    ds["data"].attrs["crs"] = "EPSG:4326"
    ds.to_zarr(str(mosaic_path), mode="w", zarr_format=3)

    features = gpd.GeoDataFrame(
        {COL_FEATURE_ID: ["a"], "geometry": [box(0, 0, 4, 4)]},
        geometry="geometry",
    )

    result_ds = zonal_stats(
        str(mosaic_path),
        features,
        transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
        crs=pyproj.CRS("EPSG:4326"),
        stats=("count", "mean"),
        array_name="data",
        decode_coords=True,
    )

    output_path = tmp_path / "stats_ts.parquet"
    result_ds.write_parquet(str(output_path))
    out = pd.read_parquet(output_path)

    assert len(out) == 6  # 3 times × 2 bands
    assert "band" in out.columns
    band_vals = sorted(out["band"].unique())
    assert band_vals == ["b1", "b2"]
    assert "time" in out.columns
    time_vals = sorted(out["time"].unique())
    assert pd.Timestamp(time_vals[0]) == pd.Timestamp("2020-01-01")
    assert pd.Timestamp(time_vals[-1]) == pd.Timestamp("2020-01-03")
