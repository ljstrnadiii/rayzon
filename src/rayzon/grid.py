from __future__ import annotations

import math
from dataclasses import dataclass

from affine import Affine


@dataclass(frozen=True)
class GridSpec:
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    chunk_sizes: tuple[int, ...]
    transform: tuple[float, float, float, float, float, float]
    crs: str
    x_dim: str
    y_dim: str

    def dim_index(self, dim: str) -> int:
        try:
            return self.dims.index(dim)
        except ValueError as exc:
            raise KeyError(f"Unknown dimension: {dim}") from exc

    @property
    def x_index(self) -> int:
        return self.dim_index(self.x_dim)

    @property
    def y_index(self) -> int:
        return self.dim_index(self.y_dim)

    def chunk_grid_shape(self) -> tuple[int, ...]:
        return tuple(math.ceil(s / c) for s, c in zip(self.shape, self.chunk_sizes, strict=True))


def chunk_id_to_slices(chunk_id: tuple[int, ...], grid: GridSpec) -> tuple[slice, ...]:
    if len(chunk_id) != len(grid.dims):
        raise ValueError("chunk_id rank must match grid dims")

    slices: list[slice] = []
    for idx, (chunk_i, dim_len, chunk_len) in enumerate(
        zip(chunk_id, grid.shape, grid.chunk_sizes, strict=True)
    ):
        if chunk_i < 0:
            raise ValueError(f"chunk index must be >=0 for axis {idx}")
        start = chunk_i * chunk_len
        stop = min(start + chunk_len, dim_len)
        if start >= dim_len:
            raise ValueError(f"chunk index out of bounds for axis {idx}")
        slices.append(slice(start, stop))
    return tuple(slices)


def chunk_pixel_bounds_xy(chunk_id: tuple[int, ...], grid: GridSpec) -> tuple[int, int, int, int]:
    slices = chunk_id_to_slices(chunk_id, grid)
    ys = slices[grid.y_index]
    xs = slices[grid.x_index]
    return xs.start or 0, ys.start or 0, xs.stop or 0, ys.stop or 0


def pixel_to_world(x: float, y: float, grid: GridSpec) -> tuple[float, float]:
    wx, wy = Affine(*grid.transform) * (x, y)
    return float(wx), float(wy)


def chunk_world_bounds(
    chunk_id: tuple[int, ...], grid: GridSpec
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = chunk_pixel_bounds_xy(chunk_id, grid)
    c00 = pixel_to_world(x0, y0, grid)
    c10 = pixel_to_world(x1, y0, grid)
    c01 = pixel_to_world(x0, y1, grid)
    c11 = pixel_to_world(x1, y1, grid)

    xs = (c00[0], c10[0], c01[0], c11[0])
    ys = (c00[1], c10[1], c01[1], c11[1])
    return min(xs), min(ys), max(xs), max(ys)


def world_bbox_to_chunk_ranges(
    bbox: tuple[float, float, float, float],
    grid: GridSpec,
) -> tuple[range, range]:
    minx, miny, maxx, maxy = bbox
    inv = ~Affine(*grid.transform)

    px0, py0 = inv * (minx, maxy)
    px1, py1 = inv * (maxx, miny)

    x_min = max(0, math.floor(min(px0, px1)))
    x_max = min(grid.shape[grid.x_index], math.ceil(max(px0, px1)))
    y_min = max(0, math.floor(min(py0, py1)))
    y_max = min(grid.shape[grid.y_index], math.ceil(max(py0, py1)))

    cx = grid.chunk_sizes[grid.x_index]
    cy = grid.chunk_sizes[grid.y_index]
    x_chunk_start = x_min // cx
    x_chunk_stop = math.ceil(x_max / cx)
    y_chunk_start = y_min // cy
    y_chunk_stop = math.ceil(y_max / cy)

    return range(y_chunk_start, y_chunk_stop), range(x_chunk_start, x_chunk_stop)


def reconstruct_grid_spec(
    *,
    dims: list[str],
    shape: list[int],
    chunk_sizes: list[int],
    transform_coeffs: list[float],
    crs: str,
    x_dim: str,
    y_dim: str,
) -> GridSpec:
    return GridSpec(
        dims=tuple(dims),
        shape=tuple(shape),
        chunk_sizes=tuple(chunk_sizes),
        transform=(
            transform_coeffs[0],
            transform_coeffs[1],
            transform_coeffs[2],
            transform_coeffs[3],
            transform_coeffs[4],
            transform_coeffs[5],
        ),
        crs=crs,
        x_dim=x_dim,
        y_dim=y_dim,
    )
