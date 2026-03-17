from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import urlparse

import numpy as np
import zarr
from obstore.store import LocalStore, S3Store
from zarr.storage import ObjectStore

from rayzon.grid import GridSpec


class ZarrBackend(StrEnum):
    ZARR_PYTHON = "zarr-python"


class ZarrArrayOpener(Protocol):
    def __call__(self, store_uri: str, *, array_name: str | None) -> zarr.Array: ...


class _ZarrPythonOpener:
    def __call__(self, store_uri: str, *, array_name: str | None) -> zarr.Array:
        store = get_obstore(store_uri)
        if array_name is not None:
            group = zarr.open_group(store=store, mode="r")
            arr = group[array_name]
            if not isinstance(arr, zarr.Array):
                raise TypeError(f"'{array_name}' in {store_uri} is not a zarr array")
            return arr
        return zarr.open_array(store=store, mode="r")


_ZARR_OPENERS: dict[ZarrBackend, ZarrArrayOpener] = {
    ZarrBackend.ZARR_PYTHON: _ZarrPythonOpener(),
}


def build_grid_spec(
    store_uri: str,
    *,
    x_dim: str,
    y_dim: str,
    transform: tuple[float, float, float, float, float, float],
    crs: str,
    array_name: str | None = None,
    zarr_backend: ZarrBackend = ZarrBackend.ZARR_PYTHON,
) -> tuple[GridSpec, np.dtype, float | None, float | None]:
    array = open_zarr_array(store_uri, array_name=array_name, zarr_backend=zarr_backend)
    shape = tuple(int(size) for size in array.shape)
    chunk_sizes = _normalize_chunk_sizes(array.chunks, shape)
    attrs = dict(array.attrs)
    raw_dims = _resolve_raw_dims(array, attrs)
    dims = _resolve_dims(raw_dims, rank=len(shape), x_dim=x_dim, y_dim=y_dim)

    grid = GridSpec(
        dims=dims,
        shape=shape,
        chunk_sizes=chunk_sizes,
        transform=transform,
        crs=crs,
        x_dim=x_dim,
        y_dim=y_dim,
    )
    pixel_dtype = resolve_pixel_dtype(array)
    scale, offset = resolve_scale_offset(array)
    return grid, pixel_dtype, scale, offset


def open_zarr_array(
    store_uri: str,
    *,
    array_name: str | None = None,
    zarr_backend: ZarrBackend = ZarrBackend.ZARR_PYTHON,
) -> zarr.Array:
    opener = _ZARR_OPENERS.get(zarr_backend)
    if opener is None:
        raise ValueError(f"Unsupported zarr backend: {zarr_backend}")
    return opener(store_uri, array_name=array_name)


def _normalize_chunk_sizes(chunks: Sequence[int] | None, shape: tuple[int, ...]) -> tuple[int, ...]:
    if chunks is None:
        return shape
    return tuple(int(size) for size in chunks)


def _resolve_dims(
    raw_dims: Sequence[str] | None,
    *,
    rank: int,
    x_dim: str,
    y_dim: str,
) -> tuple[str, ...]:
    if raw_dims is not None:
        dims = tuple(str(value) for value in raw_dims)
        if len(dims) != rank:
            raise ValueError("_ARRAY_DIMENSIONS rank does not match array rank")
    else:
        default_dims = [f"dim_{i}" for i in range(rank)]
        if rank >= 2:
            default_dims[-2] = y_dim
            default_dims[-1] = x_dim
        dims = tuple(default_dims)

    if x_dim not in dims or y_dim not in dims:
        raise ValueError("Configured x_dim/y_dim must exist in zarr dimensions")
    return dims


def _resolve_raw_dims(
    array: zarr.Array,
    attrs: Mapping[str, Any],
) -> Sequence[str] | None:
    metadata = getattr(array, "metadata", None)
    if metadata is not None:
        dimension_names = getattr(metadata, "dimension_names", None)
        if isinstance(dimension_names, tuple | list):
            return dimension_names
    attr_dims = attrs.get("_ARRAY_DIMENSIONS")
    if isinstance(attr_dims, list | tuple):
        return tuple(str(value) for value in attr_dims)
    return None


def resolve_pixel_dtype(array: zarr.Array) -> np.dtype:
    stored = array.dtype
    attrs = dict(array.attrs)
    scale = attrs.get("scale_factor")
    offset = attrs.get("add_offset")
    if scale is None and offset is None:
        return np.dtype(stored)
    # Mimic xarray: result_type of stored dtype with the scale/offset values.
    parts: list[np.dtype] = [np.dtype(stored)]
    if scale is not None:
        parts.append(np.result_type(float(scale)))  # type: ignore[arg-type]
    if offset is not None:
        parts.append(np.result_type(float(offset)))  # type: ignore[arg-type]
    return np.result_type(*parts)


def resolve_scale_offset(array: zarr.Array) -> tuple[float | None, float | None]:
    attrs = dict(array.attrs)
    raw_scale = attrs.get("scale_factor")
    raw_offset = attrs.get("add_offset")
    scale = float(cast("int | float", raw_scale)) if raw_scale is not None else None
    offset = float(cast("int | float", raw_offset)) if raw_offset is not None else None
    return scale, offset


def resolve_dim_coords(
    store_uri: str,
    *,
    dims: Sequence[str],
    x_dim: str,
    y_dim: str,
    array_name: str | None = None,
) -> dict[str, list]:
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "decode_coords requires xarray. "
            "Install with the 'xarray' extra, for example "
            '`pip install "rayzon[xarray]"` or '
            '`uv pip install "rayzon[xarray]"`.'
        ) from exc

    non_spatial = [d for d in dims if d not in (x_dim, y_dim)]
    if not non_spatial:
        return {}

    if array_name is not None:
        ds = xr.open_dataset(store_uri, engine="zarr", chunks=None)
        da = ds[array_name]
    else:
        da = xr.open_dataarray(store_uri, engine="zarr", chunks=None)

    coords: dict[str, list] = {}
    for dim in non_spatial:
        if dim in da.coords:
            coords[dim] = da.coords[dim].values.tolist()
        else:
            coords[dim] = list(range(da.sizes.get(dim, 0)))
    return coords


def get_obstore(mosaic_uri: str, *, region: str | None = None) -> ObjectStore:
    parsed = urlparse(mosaic_uri)
    scheme = parsed.scheme.lower()

    if scheme in ("", "file"):
        local_path = parsed.path if scheme == "file" else mosaic_uri
        return ObjectStore(LocalStore(prefix=Path(local_path)))

    if scheme == "s3":
        if not parsed.netloc:
            raise ValueError(f"Invalid S3 URI: {mosaic_uri}")

        resolved_region = (
            region
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )

        return ObjectStore(
            S3Store(
                bucket=parsed.netloc,
                prefix=parsed.path.lstrip("/"),
                config={"region": resolved_region},
            )
        )

    raise ValueError(f"Unsupported store URI scheme for mosaic: {mosaic_uri}")
