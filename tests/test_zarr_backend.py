from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import xarray as xr
import zarr
from zarr.storage import ObjectStore

from rayzon.zarr_backend import ZarrBackend, build_grid_spec, get_obstore


def test_build_grid_spec_standalone_array(tmp_path: Path) -> None:
    path = tmp_path / "mosaic-array.zarr"

    array = zarr.open_array(
        str(path),
        mode="w",
        zarr_format=3,
        shape=(10, 100, 100),
        chunks=(5, 10, 10),
        dtype=np.float32,
        dimension_names=("time", "y", "x"),
    )
    array.attrs["transform"] = [1.0, 0.0, 0.0, 0.0, -1.0, 100.0]
    array.attrs["crs"] = "EPSG:4326"

    grid, pixel_dtype, _, _ = build_grid_spec(
        str(path),
        x_dim="x",
        y_dim="y",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 100.0),
        crs="EPSG:4326",
    )
    assert grid.dims == ("time", "y", "x")
    assert grid.shape == (10, 100, 100)
    assert grid.chunk_sizes == (5, 10, 10)
    assert grid.crs == "EPSG:4326"
    assert pixel_dtype == np.float32


def test_build_grid_spec_group_with_array_name(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "mosaic-group.zarr"), mode="w", zarr_format=3)
    root.create_array(
        "data",
        shape=(100, 100),
        chunks=(10, 10),
        dtype=np.float32,
        dimension_names=("y", "x"),
    )

    grid, *_ = build_grid_spec(
        str(tmp_path / "mosaic-group.zarr"),
        x_dim="x",
        y_dim="y",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 100.0),
        crs="EPSG:4326",
        array_name="data",
    )
    assert grid.dims == ("y", "x")
    assert grid.chunk_sizes == (10, 10)


def test_get_obstore_local(tmp_path: Path) -> None:
    local_path = tmp_path / "local.zarr"
    zarr.open_group(str(local_path), mode="w")
    store = get_obstore(str(local_path))
    assert isinstance(store, ObjectStore)


def test_get_obstore_s3_uri() -> None:
    store = get_obstore("s3://example-bucket/path/to/mosaic.zarr", region="us-west-2")
    assert isinstance(store, ObjectStore)


def test_get_obstore_s3_uri_accepts_anon_storage_option() -> None:
    store = get_obstore(
        "s3://example-bucket/path/to/mosaic.zarr",
        storage_options={"anon": True, "region": "us-west-2"},
    )
    assert isinstance(store, ObjectStore)


def test_build_grid_spec_from_xarray_written_zarr_v3(tmp_path: Path) -> None:
    path = tmp_path / "xarray-mosaic.zarr"
    data = xr.DataArray(
        np.ones((3, 20, 30), dtype=np.float32),
        dims=("time", "y", "x"),
        name="data",
    ).to_dataset()

    data.to_zarr(str(path), mode="w", zarr_format=3)

    grid, *_ = build_grid_spec(
        str(path),
        x_dim="x",
        y_dim="y",
        array_name="data",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 20.0),
        crs="EPSG:4326",
    )
    assert grid.dims == ("time", "y", "x")
    assert grid.shape == (3, 20, 30)


def test_build_grid_spec_rejects_unknown_backend(tmp_path: Path) -> None:
    path = tmp_path / "mosaic-array.zarr"
    zarr.open_array(
        str(path),
        mode="w",
        zarr_format=3,
        shape=(10, 10),
        chunks=(5, 5),
        dtype=np.float32,
        dimension_names=("y", "x"),
    )

    with pytest.raises(ValueError, match="Unsupported zarr backend"):
        build_grid_spec(
            str(path),
            x_dim="x",
            y_dim="y",
            transform=(1.0, 0.0, 0.0, 0.0, -1.0, 10.0),
            crs="EPSG:4326",
            zarr_backend=cast(ZarrBackend, "unknown"),
        )
