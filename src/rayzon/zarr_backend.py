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
from zarr.errors import ContainsArrayError
from zarr.storage import ObjectStore

from rayzon.grid import GridSpec


class ZarrBackend(StrEnum):
    ZARR_PYTHON = "zarr-python"


class ZarrArrayOpener(Protocol):
    def __call__(
        self,
        store_uri: str,
        *,
        array_name: str | None,
        storage_options: Mapping[str, Any] | None = None,
    ) -> zarr.Array: ...


class _ZarrPythonOpener:
    def __call__(
        self,
        store_uri: str,
        *,
        array_name: str | None,
        storage_options: Mapping[str, Any] | None = None,
    ) -> zarr.Array:
        store = get_obstore(store_uri, storage_options=storage_options)
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
    transform: tuple[float, float, float, float, float, float] | None,
    crs: str | None,
    array_name: str | None = None,
    zarr_backend: ZarrBackend = ZarrBackend.ZARR_PYTHON,
    storage_options: Mapping[str, Any] | None = None,
) -> tuple[GridSpec, np.dtype, float | None, float | None]:
    array = open_zarr_array(
        store_uri,
        array_name=array_name,
        zarr_backend=zarr_backend,
        storage_options=storage_options,
    )
    shape = tuple(int(size) for size in array.shape)
    chunks = array.shards or array.chunks
    chunk_sizes = _normalize_chunk_sizes(chunks, shape)
    attrs = dict(array.attrs)
    raw_dims = _resolve_raw_dims(array, attrs)
    dims = _resolve_dims(raw_dims, rank=len(shape), x_dim=x_dim, y_dim=y_dim)
    resolved_transform, resolved_crs = resolve_geospatial_metadata(
        store_uri,
        array_name=array_name,
        array_attrs=attrs,
        transform=transform,
        crs=crs,
        storage_options=storage_options,
    )

    grid = GridSpec(
        dims=dims,
        shape=shape,
        chunk_sizes=chunk_sizes,
        transform=resolved_transform,
        crs=resolved_crs,
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
    storage_options: Mapping[str, Any] | None = None,
) -> zarr.Array:
    opener = _ZARR_OPENERS.get(zarr_backend)
    if opener is None:
        raise ValueError(f"Unsupported zarr backend: {zarr_backend}")
    return opener(store_uri, array_name=array_name, storage_options=storage_options)


def open_zarr_group(
    store_uri: str,
    *,
    storage_options: Mapping[str, Any] | None = None,
) -> zarr.Group:
    store = get_obstore(store_uri, storage_options=storage_options)
    return zarr.open_group(store=store, mode="r")


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
    storage_options: Mapping[str, Any] | None = None,
) -> dict[str, list]:
    non_spatial = [d for d in dims if d not in (x_dim, y_dim)]
    if not non_spatial:
        return {}

    dim_sizes = _resolve_dim_sizes(
        store_uri,
        dims=dims,
        array_name=array_name,
        storage_options=storage_options,
    )
    coords: dict[str, list] = {}
    try:
        group = open_zarr_group(store_uri, storage_options=storage_options)
    except ContainsArrayError:
        return {dim: list(range(dim_sizes.get(dim, 0))) for dim in non_spatial}
    for dim in non_spatial:
        if dim in group and isinstance(group[dim], zarr.Array):
            coords[dim] = _decode_coord_values(np.asarray(group[dim]), dict(group[dim].attrs))
        else:
            coords[dim] = list(range(dim_sizes.get(dim, 0)))
    return coords


def _resolve_dim_sizes(
    store_uri: str,
    *,
    dims: Sequence[str],
    array_name: str | None,
    storage_options: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    array = open_zarr_array(
        store_uri,
        array_name=array_name,
        storage_options=storage_options,
    )
    return {dim: int(size) for dim, size in zip(dims, array.shape, strict=True)}


def _decode_coord_values(values: np.ndarray, attrs: Mapping[str, Any]) -> list:
    units = attrs.get("units")
    if isinstance(units, str) and "since" in units:
        try:
            from xarray.coding.times import decode_cf_datetime

            calendar = attrs.get("calendar")
            decoded = decode_cf_datetime(values, units, calendar=calendar)
            if np.issubdtype(decoded.dtype, np.datetime64):
                return cast("list[Any]", decoded.astype("datetime64[us]").tolist())
            return cast("list[Any]", decoded.tolist())
        except Exception:
            pass
    return cast("list[Any]", values.tolist())


def get_obstore(
    mosaic_uri: str,
    *,
    region: str | None = None,
    storage_options: Mapping[str, Any] | None = None,
) -> ObjectStore:
    parsed = urlparse(mosaic_uri)
    scheme = parsed.scheme.lower()

    if scheme in ("", "file"):
        local_path = parsed.path if scheme == "file" else mosaic_uri
        return ObjectStore(LocalStore(prefix=Path(local_path)))

    if scheme == "s3":
        if not parsed.netloc:
            raise ValueError(f"Invalid S3 URI: {mosaic_uri}")

        options = _normalize_storage_options(storage_options)
        resolved_region = (
            cast("str | None", options.pop("region", None))
            or region
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )

        s3_config = dict(cast("Mapping[str, Any]", options.pop("config", {})))
        s3_config.setdefault("region", resolved_region)
        return ObjectStore(
            S3Store(
                bucket=parsed.netloc,
                prefix=parsed.path.lstrip("/"),
                config=cast("Any", s3_config),
                **options,
            )
        )

    raise ValueError(f"Unsupported store URI scheme for mosaic: {mosaic_uri}")


def resolve_geospatial_metadata(
    store_uri: str,
    *,
    array_name: str | None,
    array_attrs: Mapping[str, Any] | None = None,
    transform: tuple[float, float, float, float, float, float] | None,
    crs: str | None,
    storage_options: Mapping[str, Any] | None = None,
) -> tuple[tuple[float, float, float, float, float, float], str]:
    group_attrs: Mapping[str, Any] = {}
    if array_name is not None:
        group_attrs = dict(open_zarr_group(store_uri, storage_options=storage_options).attrs)
    attrs = dict(array_attrs or {})

    resolved_transform = transform or _transform_from_attrs(attrs, group_attrs)
    if resolved_transform is None:
        raise ValueError(
            "Could not determine raster transform. Pass transform=Affine(...) or set "
            "'transform'/'spatial:transform' in the array or root group attrs."
        )

    resolved_crs = crs or _crs_from_attrs(attrs, group_attrs)
    if resolved_crs is None:
        raise ValueError(
            "Could not determine raster CRS. Pass crs=pyproj.CRS(...) or set "
            "'crs'/'proj:code'/'proj:wkt2' in the array or root group attrs."
        )

    return resolved_transform, resolved_crs


def _transform_from_attrs(
    array_attrs: Mapping[str, Any],
    group_attrs: Mapping[str, Any],
) -> tuple[float, float, float, float, float, float] | None:
    for attrs in (array_attrs, group_attrs):
        raw = attrs.get("transform") or attrs.get("spatial:transform")
        if isinstance(raw, Sequence) and len(raw) == 6:
            values = tuple(float(value) for value in raw)
            return cast("tuple[float, float, float, float, float, float]", values)
    return None


def _crs_from_attrs(
    array_attrs: Mapping[str, Any],
    group_attrs: Mapping[str, Any],
) -> str | None:
    for attrs in (array_attrs, group_attrs):
        for key in ("crs", "proj:code", "proj:wkt2"):
            raw = attrs.get(key)
            if raw is None:
                continue
            if key == "proj:code" and isinstance(raw, int):
                return f"EPSG:{raw}"
            return str(raw)
    return None


def _normalize_storage_options(
    storage_options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    options = dict(storage_options or {})
    if "anon" in options and "skip_signature" not in options:
        options["skip_signature"] = bool(options.pop("anon"))
    else:
        options.pop("anon", None)
    return options
