from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol, cast

import numpy as np
from affine import Affine
from numpy.typing import NDArray
from rasterio.features import rasterize
from shapely.geometry.base import BaseGeometry


class RasterizeBackend(StrEnum):
    RASTERIO = "rasterio"


class RasterizeWindowBackend(Protocol):
    def __call__(
        self,
        geometry: BaseGeometry,
        out_shape: tuple[int, int],
        transform: Affine | tuple[float, ...] | list[float] | NDArray[np.floating],
        all_touched: bool = False,
    ) -> NDArray[np.bool_]: ...


class _RasterioWindowBackend:
    def __call__(
        self,
        geometry: BaseGeometry,
        out_shape: tuple[int, int],
        transform: Affine | tuple[float, ...] | list[float] | NDArray[np.floating],
        all_touched: bool = False,
    ) -> NDArray[np.bool_]:
        affine_transform = _coerce_affine(transform)
        mask = rasterize(
            [(geometry, 1)],
            out_shape=out_shape,
            transform=affine_transform,
            fill=0,
            all_touched=all_touched,
            dtype="uint8",
        )
        return cast(NDArray[np.bool_], np.asarray(mask, dtype=bool))


def rasterize_geometry_window(
    geometry: BaseGeometry,
    out_shape: tuple[int, int],
    transform: Affine | tuple[float, ...] | list[float] | NDArray[np.floating],
    all_touched: bool = False,
    rasterize_backend: RasterizeBackend = RasterizeBackend.RASTERIO,
) -> NDArray[np.bool_]:
    match rasterize_backend:
        case RasterizeBackend.RASTERIO:
            return _RasterioWindowBackend()(
                geometry,
                out_shape,
                transform,
                all_touched=all_touched,
            )
        case _:
            raise ValueError(f"Unsupported rasterize backend: {rasterize_backend}")


def _coerce_affine(
    transform: Affine | tuple[float, ...] | list[float] | NDArray[np.floating],
) -> Affine:
    if isinstance(transform, Affine):
        return transform

    values: list[float]
    if isinstance(transform, np.ndarray):
        if transform.size != 6:
            raise ValueError("Transform numpy array must contain exactly 6 values")
        values = [float(v) for v in transform.reshape(-1).tolist()]
    elif isinstance(transform, Sequence):
        if len(transform) != 6:
            raise ValueError("Transform sequence must contain exactly 6 values")
        values = [float(v) for v in transform]
    else:
        raise TypeError(f"Unsupported transform type: {type(transform)!r}")

    return Affine(*values)
