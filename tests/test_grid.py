from affine import Affine

from rzzs.grid import (
    GridSpec,
    chunk_id_to_slices,
    chunk_pixel_bounds_xy,
    world_bbox_to_chunk_ranges,
)


def _grid() -> GridSpec:
    return GridSpec(
        dims=("time", "band", "y", "x"),
        shape=(10, 2, 100, 100),
        chunk_sizes=(5, 1, 10, 10),
        transform=Affine(1, 0, 0, 0, -1, 100),
        crs="EPSG:4326",
        x_dim="x",
        y_dim="y",
    )


def test_chunk_id_to_slices() -> None:
    slices = chunk_id_to_slices((0, 0, 2, 3), _grid())
    assert slices[2].start == 20
    assert slices[2].stop == 30
    assert slices[3].start == 30
    assert slices[3].stop == 40


def test_chunk_pixel_bounds_xy() -> None:
    bounds = chunk_pixel_bounds_xy((0, 0, 2, 3), _grid())
    assert bounds == (30, 20, 40, 30)


def test_world_bbox_to_chunk_ranges() -> None:
    y_range, x_range = world_bbox_to_chunk_ranges((20, 60, 40, 80), _grid())
    assert list(x_range) == [2, 3]
    assert list(y_range) == [2, 3]
