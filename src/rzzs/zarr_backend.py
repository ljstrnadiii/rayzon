from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import zarr
from affine import Affine
from obstore.store import LocalStore, S3Store
from zarr.storage import ObjectStore

from rzzs.grid import GridSpec


class ZarrBackend(StrEnum):
    ZARR_PYTHON = "zarr-python"


class ZarrArrayBackend(Protocol):
    def __call__(self, mosaic_uri: str, *, array_path: str | None) -> zarr.Array: ...


class _ZarrPythonArrayBackend:
    def __call__(self, mosaic_uri: str, *, array_path: str | None) -> zarr.Array:
        return _open_zarr_python_array(mosaic_uri, array_path=array_path)


_ARRAY_BACKENDS: dict[ZarrBackend, ZarrArrayBackend] = {
    ZarrBackend.ZARR_PYTHON: _ZarrPythonArrayBackend(),
}


def build_grid_spec_from_mosaic(
    mosaic_uri: str,
    *,
    x_dim: str,
    y_dim: str,
    array_path: str | None = None,
    dst_crs: str | None = None,
    zarr_backend: ZarrBackend = ZarrBackend.ZARR_PYTHON,
) -> GridSpec:
    array = _open_mosaic_array(
        mosaic_uri,
        array_path=array_path,
        zarr_backend=zarr_backend,
    )

    shape = tuple(int(size) for size in array.shape)
    chunk_sizes = _normalize_chunk_sizes(array.chunks, shape)
    attrs = dict(array.attrs)
    raw_dims = _resolve_raw_dims(array, attrs)

    dims = _resolve_dims(raw_dims, rank=len(shape), x_dim=x_dim, y_dim=y_dim)
    transform = _resolve_transform(attrs.get("transform"))
    crs = dst_crs or _resolve_crs(attrs.get("crs"))

    return GridSpec(
        dims=dims,
        shape=shape,
        chunk_sizes=chunk_sizes,
        transform=transform,
        crs=crs,
        x_dim=x_dim,
        y_dim=y_dim,
    )


def _open_mosaic_array(
    mosaic_uri: str,
    *,
    array_path: str | None,
    zarr_backend: ZarrBackend,
) -> zarr.Array:
    backend = _ARRAY_BACKENDS.get(zarr_backend)
    if backend is None:
        raise ValueError(f"Unsupported zarr backend: {zarr_backend}")
    return backend(mosaic_uri, array_path=array_path)


def _open_zarr_python_array(mosaic_uri: str, *, array_path: str | None) -> zarr.Array:
    store = get_obstore(mosaic_uri)

    if array_path:
        group = zarr.open_group(store=store, mode="r")
        return _as_array(group[array_path])

    try:
        return zarr.open_array(store=store, mode="r")
    except Exception as exc:
        group = zarr.open_group(store=store, mode="r")
        keys = list(group.array_keys())
        if not keys:
            raise ValueError(f"No arrays found in zarr group at {mosaic_uri}") from exc
        if len(keys) > 1 and "data" not in keys:
            raise ValueError(
                "Mosaic zarr group contains multiple arrays; provide array_path to disambiguate"
            ) from exc
        key = "data" if "data" in keys else keys[0]
        return _as_array(group[key])


def _normalize_chunk_sizes(chunks: Sequence[int] | None, shape: tuple[int, ...]) -> tuple[int, ...]:
    if chunks is None:
        return shape
    return tuple(int(size) for size in chunks)


def _resolve_dims(
    raw_dims: object,
    *,
    rank: int,
    x_dim: str,
    y_dim: str,
) -> tuple[str, ...]:
    if isinstance(raw_dims, list | tuple):
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


def _resolve_raw_dims(array: zarr.Array, attrs: Mapping[str, object]) -> object:
    metadata = getattr(array, "metadata", None)
    if metadata is not None:
        dimension_names = getattr(metadata, "dimension_names", None)
        if isinstance(dimension_names, tuple | list):
            return dimension_names
    return attrs.get("_ARRAY_DIMENSIONS")


def _resolve_transform(raw_transform: object) -> Affine:
    if isinstance(raw_transform, list | tuple) and len(raw_transform) == 6:
        values = tuple(float(value) for value in raw_transform)
        return Affine(*values)
    return Affine.identity()


def _resolve_crs(raw_crs: object) -> str | None:
    if raw_crs is None:
        return None
    return str(raw_crs)


def _as_array(value: object) -> zarr.Array:
    if isinstance(value, zarr.Array):
        return value
    raise TypeError("Resolved zarr node is not an array")


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
