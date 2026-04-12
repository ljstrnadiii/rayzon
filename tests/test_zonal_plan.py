from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import zarr
from affine import Affine
from shapely.geometry import box

from rayzon.grid import GridSpec
from rayzon.types import COL_FEATURE_ID
from rayzon.zonal_plan import (
    _resolve_allowed_chunk_ids_by_dim,
    _resolve_allowed_dim_indices,
    _resolve_coord_selection_with_xarray,
    _resolve_dim_selector_indices,
    build_zonal_stats_plan,
)


def test_build_zonal_stats_plan_derives_expected_fields(tmp_path: Path) -> None:
    mosaic_path = tmp_path / "plan_mosaic.zarr"
    array = zarr.open_array(
        str(mosaic_path),
        mode="w",
        zarr_format=3,
        shape=(2, 3, 4, 4),
        chunks=(1, 1, 4, 4),
        dtype=np.float32,
        dimension_names=("time", "band", "y", "x"),
    )
    array[:] = np.ones((2, 3, 4, 4), dtype=np.float32)
    array.attrs["transform"] = [1.0, 0.0, 0.0, 0.0, -1.0, 4.0]
    array.attrs["crs"] = "EPSG:4326"

    features = gpd.GeoDataFrame(
        {COL_FEATURE_ID: ["a"], "geometry": [box(0, 0, 4, 4)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    plan = build_zonal_stats_plan(
        str(mosaic_path),
        features,
        transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
        crs=pyproj.CRS("EPSG:4326"),
        stats=("count", "mean"),
        selectors={"time": 1},
    )

    assert plan.non_spatial_dims == ["time", "band"]
    assert plan.partial_columns == ("_partial_count", "_partial_n_valid", "_partial_sum")
    assert plan.allowed_dim_indices["time"] == [1]
    assert plan.allowed_chunk_ids_by_dim["time"] == [1]
    assert plan.group_keys == [COL_FEATURE_ID, "time", "band"]
    assert "allowed_chunk_ids_by_dim" in plan.feature_map_kwargs
    assert plan.chunk_kwargs["feature_id_pa_type"] is not None


def _grid() -> GridSpec:
    return GridSpec(
        dims=("time", "band", "y", "x"),
        shape=(4, 3, 8, 8),
        chunk_sizes=(2, 1, 4, 4),
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 8.0),
        crs="EPSG:4326",
        x_dim="x",
        y_dim="y",
    )


def test_resolve_dim_selector_indices_with_integer() -> None:
    indices = _resolve_dim_selector_indices(
        coord_values=list(range(4)),
        dim_size=4,
        selector=2,
        dim_name="time",
    )

    assert indices == [2]


def test_resolve_dim_selector_indices_with_integer_slice() -> None:
    indices = _resolve_dim_selector_indices(
        coord_values=list(range(6)),
        dim_size=6,
        selector=slice(1, 5, 2),
        dim_name="time",
    )

    assert indices == [1, 3]


def test_resolve_coord_selection_with_xarray_datetime_string() -> None:
    coord_values = list(pd.to_datetime(["2022-12-31", "2023-01-01", "2023-06-01", "2024-01-01"]))

    indices = _resolve_coord_selection_with_xarray(
        coord_values=coord_values,
        selector="2023",
        dim_name="time",
    )

    assert indices == [1, 2]


def test_resolve_allowed_dim_indices_and_chunk_ids() -> None:
    grid = _grid()
    dim_coord_values = {
        "time": list(pd.to_datetime(["2022-12-31", "2023-01-01", "2023-06-01", "2024-01-01"])),
        "band": ["b1", "b2", "b3"],
    }

    allowed_dim_indices = _resolve_allowed_dim_indices(
        grid_spec=grid,
        non_spatial_dims=["time", "band"],
        dim_coord_values=dim_coord_values,
        selectors={"time": "2023", "band": "b2"},
    )
    allowed_chunk_ids = _resolve_allowed_chunk_ids_by_dim(
        grid_spec=grid,
        allowed_dim_indices=allowed_dim_indices,
    )

    assert allowed_dim_indices == {"time": [1, 2], "band": [1]}
    assert allowed_chunk_ids == {"time": [0, 1], "band": [1]}
