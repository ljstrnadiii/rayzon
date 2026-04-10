from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import zarr
from affine import Affine
from shapely.geometry import box

from rayzon.pipeline import zonal_stats
from rayzon.types import COL_FEATURE_ID


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


def test_zonal_stats_selectors_subset_time_dimension(tmp_path: Path) -> None:
    import xarray as xr

    mosaic_path = tmp_path / "mosaic_selector.zarr"
    times = pd.to_datetime(["2022-12-31", "2023-01-01", "2023-06-01", "2024-01-01"])
    ds = xr.Dataset(
        {"data": (("time", "band", "y", "x"), np.ones((4, 1, 4, 4), dtype=np.float32))},
        coords={
            "time": times,
            "band": ["b1"],
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
        selectors={"time": "2023"},
    )

    output_path = tmp_path / "stats_selector.parquet"
    result_ds.write_parquet(str(output_path))
    out = pd.read_parquet(output_path).sort_values("time").reset_index(drop=True)

    assert len(out) == 2
    assert sorted(pd.Timestamp(value) for value in out["time"].unique()) == [
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-06-01"),
    ]
    assert out["count"].tolist() == [16, 16]


def test_zonal_stats_infers_root_spatial_metadata_for_group_arrays(
    tmp_path: Path,
) -> None:
    import xarray as xr

    mosaic_path = tmp_path / "mosaic_vectors.zarr"
    times = pd.date_range("2020-01-01", periods=2, freq="YS")
    bands = ["b1", "b2", "b3"]
    data = np.stack(
        [
            np.stack([np.full((4, 4), fill_value=t * 10 + b, dtype=np.float32) for b in range(3)])
            for t in range(2)
        ]
    )
    ds = xr.Dataset(
        {"embeddings": (("time", "band", "y", "x"), data)},
        coords={
            "time": times,
            "band": bands,
            "y": np.arange(4.0, 0.0, -1.0),
            "x": np.arange(0.0, 4.0),
        },
    )
    ds.attrs["spatial:transform"] = [1.0, 0.0, 0.0, 0.0, -1.0, 4.0]
    ds.attrs["proj:code"] = "EPSG:4326"
    ds.to_zarr(str(mosaic_path), mode="w", zarr_format=3)

    features = gpd.GeoDataFrame(
        {COL_FEATURE_ID: ["a"], "geometry": [box(0, 0, 4, 4)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    result_ds = zonal_stats(
        str(mosaic_path),
        features,
        stats=("mean",),
        array_name="embeddings",
        decode_coords=True,
    )

    output_path = tmp_path / "stats_group_attrs.parquet"
    result_ds.write_parquet(str(output_path))
    out = pd.read_parquet(output_path).sort_values("time").reset_index(drop=True)

    assert len(out) == 6
    assert set(out["band"]) == set(bands)
    assert out.iloc[0][COL_FEATURE_ID] == "a"
    assert set(pd.Timestamp(value) for value in out["time"].unique()) == {
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2021-01-01"),
    }

    expected = {
        (pd.Timestamp("2020-01-01"), "b1"): 0.0,
        (pd.Timestamp("2020-01-01"), "b2"): 1.0,
        (pd.Timestamp("2020-01-01"), "b3"): 2.0,
        (pd.Timestamp("2021-01-01"), "b1"): 10.0,
        (pd.Timestamp("2021-01-01"), "b2"): 11.0,
        (pd.Timestamp("2021-01-01"), "b3"): 12.0,
    }
    for _, row in out.iterrows():
        key = (pd.Timestamp(row["time"]), row["band"])
        assert row["mean"] == expected[key]


def test_zonal_stats_quantiles_tdigest_approximation(tmp_path: Path) -> None:
    mosaic_path = tmp_path / "mosaic_quantiles.zarr"
    values = np.concatenate(
        [
            np.linspace(0.0, 10.0, 900, dtype=np.float32),
            np.linspace(100.0, 1000.0, 100, dtype=np.float32),
        ]
    )
    array = zarr.open_array(
        str(mosaic_path),
        mode="w",
        zarr_format=3,
        shape=(40, 25),
        chunks=(20, 25),
        dtype=np.float32,
        dimension_names=("y", "x"),
    )
    array[:] = values.reshape(40, 25)
    array.attrs["transform"] = [1.0, 0.0, 0.0, 0.0, -1.0, 40.0]
    array.attrs["crs"] = "EPSG:4326"

    features = gpd.GeoDataFrame(
        {COL_FEATURE_ID: ["a"], "geometry": [box(0.0, 0.0, 25.0, 40.0)]},
        geometry="geometry",
    )

    result_ds = zonal_stats(
        str(mosaic_path),
        features,
        transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 40.0),
        crs=pyproj.CRS("EPSG:4326"),
        stats=("p50", "p95"),
    )

    output_path = tmp_path / "stats_quantiles.parquet"
    result_ds.write_parquet(str(output_path))
    out = pd.read_parquet(output_path)
    assert len(out) == 1
    assert {"p50", "p95"} <= set(out.columns)

    expected_p50 = float(np.quantile(values.astype(np.float64), 0.5))
    expected_p95 = float(np.quantile(values.astype(np.float64), 0.95))
    actual_p50 = float(out.iloc[0]["p50"])
    actual_p95 = float(out.iloc[0]["p95"])

    # Explicit relative tolerances to keep approximation expectations readable.
    p50_rtol = 0.0001
    p95_rtol = 0.01

    np.testing.assert_allclose(actual_p50, expected_p50, rtol=p50_rtol, atol=0.0)
    np.testing.assert_allclose(actual_p95, expected_p95, rtol=p95_rtol, atol=0.0)


def test_zonal_stats_merges_partial_rows_across_spatial_chunks(tmp_path: Path) -> None:
    import xarray as xr

    mosaic_path = tmp_path / "mosaic_cross_chunk.zarr"
    data = np.array(
        [
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ],
                [
                    [101.0, 102.0, 103.0, 104.0],
                    [105.0, 106.0, 107.0, 108.0],
                    [109.0, 110.0, 111.0, 112.0],
                    [113.0, 114.0, 115.0, 116.0],
                ],
            ]
        ],
        dtype=np.float32,
    )
    ds = xr.Dataset(
        {"embeddings": (("time", "band", "y", "x"), data)},
        coords={
            "time": pd.to_datetime(["2024-01-01"]),
            "band": ["A00", "A01"],
            "y": np.arange(4.0, 0.0, -1.0),
            "x": np.arange(0.0, 4.0),
        },
    )
    ds["embeddings"].attrs["transform"] = [1.0, 0.0, 0.0, 0.0, -1.0, 4.0]
    ds["embeddings"].attrs["crs"] = "EPSG:4326"
    ds.to_zarr(
        str(mosaic_path),
        mode="w",
        zarr_format=3,
        encoding={"embeddings": {"chunks": (1, 2, 2, 4)}},
    )

    features = gpd.GeoDataFrame(
        {COL_FEATURE_ID: ["a"], "geometry": [box(0.0, 0.0, 4.0, 4.0)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    result_ds = zonal_stats(
        str(mosaic_path),
        features,
        stats=("mean",),
        array_name="embeddings",
        decode_coords=True,
    )

    output_path = tmp_path / "stats_cross_chunk.parquet"
    result_ds.write_parquet(str(output_path))
    out = pd.read_parquet(output_path).sort_values("band").reset_index(drop=True)

    assert len(out) == 2
    assert out[COL_FEATURE_ID].tolist() == ["a", "a"]
    assert [pd.Timestamp(value) for value in out["time"]] == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-01"),
    ]
    np.testing.assert_allclose(out["mean"].to_numpy(dtype=np.float64), [8.5, 108.5])
