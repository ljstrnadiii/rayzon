from pathlib import Path

import numpy as np
import zarr
from shapely.geometry import box

from rzzs.chunk_processor import process_chunk
from rzzs.zarr_backend import build_grid_spec, open_zarr_array


def test_process_chunk_emits_partial_rows_for_2d_chunk(tmp_path: Path) -> None:
    mosaic_path = tmp_path / "mosaic2d.zarr"
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

    grid, *_ = build_grid_spec(
        str(mosaic_path),
        x_dim="x",
        y_dim="y",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
        crs="EPSG:4326",
    )
    arr = open_zarr_array(str(mosaic_path))
    rows = process_chunk(
        chunk_id=(0, 0),
        feature_ids=["f1"],
        array=arr,
        geometry_store={"f1": box(1.0, 1.0, 3.0, 3.0)},
        grid_spec=grid,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["feature_id"] == "f1"
    assert row["dim_values"] == ()
    assert isinstance(row["pixels"], np.ndarray)
    assert len(row["pixels"]) > 0


def test_process_chunk_emits_one_row_per_non_spatial_index(tmp_path: Path) -> None:
    mosaic_path = tmp_path / "mosaic3d.zarr"
    array = zarr.open_array(
        str(mosaic_path),
        mode="w",
        zarr_format=3,
        shape=(2, 4, 4),
        chunks=(2, 4, 4),
        dtype=np.float32,
        dimension_names=("time", "y", "x"),
    )
    array[:] = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
    array.attrs["transform"] = [1.0, 0.0, 0.0, 0.0, -1.0, 4.0]
    array.attrs["crs"] = "EPSG:4326"

    grid, *_ = build_grid_spec(
        str(mosaic_path),
        x_dim="x",
        y_dim="y",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
        crs="EPSG:4326",
    )
    arr = open_zarr_array(str(mosaic_path))
    rows = process_chunk(
        chunk_id=(0, 0, 0),
        feature_ids=["f1"],
        array=arr,
        geometry_store={"f1": box(1.0, 1.0, 3.0, 3.0)},
        grid_spec=grid,
    )

    assert len(rows) == 2
    dim_values = sorted(row["dim_values"] for row in rows)
    assert dim_values == [(0,), (1,)]
