from __future__ import annotations

from collections.abc import Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import ray.data

from rayzon.logging_utils import get_logger

logger = get_logger(__name__)

COORD_COL_PREFIX = "__"
COORD_COL_SUFFIX = "_coord"

VALID_FREQUENCIES = frozenset({"Y", "M", "D"})


def coord_col_name(dim: str) -> str:
    return f"{COORD_COL_PREFIX}{dim}{COORD_COL_SUFFIX}"


def is_coord_col(name: str) -> bool:
    return name.startswith(COORD_COL_PREFIX) and name.endswith(COORD_COL_SUFFIX)


def normalize_coord_columns(
    coord_columns: Mapping[str, str | None] | Sequence[str] | None,
) -> dict[str, str | None]:
    if coord_columns is None:
        return {}
    if isinstance(coord_columns, Mapping):
        out = dict(coord_columns)
    else:
        out = {col: None for col in coord_columns}
    for dim, freq in out.items():
        if freq is not None and freq not in VALID_FREQUENCIES:
            raise ValueError(
                f"Invalid frequency {freq!r} for coord_column {dim!r}. "
                f"Must be one of {sorted(VALID_FREQUENCIES)} or None."
            )
    return out


def truncate_arrow_column(
    col: pa.Array,
    freq: str | None,
) -> pa.Array:
    if freq is None:
        if pa.types.is_timestamp(col.type):
            return col.cast(pa.int64())
        return col
    if freq == "Y":
        return _temporal_to_year(col)
    if freq == "M":
        return _temporal_to_year_month(col)
    if freq == "D":
        return _temporal_to_year_month_day(col)
    raise ValueError(f"Unsupported frequency: {freq!r}")


def truncate_coord_value(value: object, freq: str | None) -> object:
    if freq is None:
        return value
    if freq == "Y":
        return _scalar_to_year(value)
    if freq == "M":
        return _scalar_to_year_month(value)
    if freq == "D":
        return _scalar_to_year_month_day(value)
    raise ValueError(f"Unsupported frequency: {freq!r}")


def add_coord_column(
    ds: ray.data.Dataset,
    dim: str,
    freq: str | None,
) -> ray.data.Dataset:
    col_name = coord_col_name(dim)

    def _add_col(batch: pa.Table, *, dim: str, freq: str | None, col_name: str) -> pa.Table:
        src = batch.column(dim)
        truncated = truncate_arrow_column(src, freq)
        return batch.append_column(pa.field(col_name, truncated.type), truncated)

    return ds.map_batches(
        _add_col,  # type: ignore[arg-type]
        fn_kwargs={"dim": dim, "freq": freq, "col_name": col_name},
        batch_format="pyarrow",
    )


def extract_unique_coord_values(
    ds: ray.data.Dataset,
    dim: str,
) -> list:
    col_name = coord_col_name(dim)
    return sorted(ds.select_columns([col_name]).unique(col_name))


def match_coord_values_to_dim_indices(
    feature_coord_values: list,
    zarr_coord_values: list,
    freq: str | None,
) -> dict[object, list[int]]:
    truncated_zarr: dict[object, list[int]] = {}
    for idx, v in enumerate(zarr_coord_values):
        key = truncate_coord_value(v, freq)
        truncated_zarr.setdefault(key, []).append(idx)

    coord_to_indices: dict[object, list[int]] = {}
    for fv in feature_coord_values:
        matches = truncated_zarr.get(fv, [])
        if matches:
            coord_to_indices[fv] = matches
        else:
            logger.warning(
                "feature coord value %r (dim=%s, freq=%s) has no matching zarr coordinate",
                fv,
                "?",
                freq,
            )
    return coord_to_indices


def build_coord_to_chunk_ids(
    coord_to_indices: dict[object, list[int]],
    chunk_size: int,
) -> dict[object, list[int]]:
    coord_to_chunks: dict[object, list[int]] = {}
    for coord_val, indices in coord_to_indices.items():
        chunks = sorted({idx // chunk_size for idx in indices})
        coord_to_chunks[coord_val] = chunks
    return coord_to_chunks


# ---------------------------------------------------------------------------
# Temporal truncation helpers
# ---------------------------------------------------------------------------


def _temporal_to_year(col: pa.Array) -> pa.Array:
    if pa.types.is_timestamp(col.type):
        return pc.year(col).cast(pa.int64())
    if pa.types.is_integer(col.type):
        return col.cast(pa.int64())
    raise TypeError(f"Cannot extract year from {col.type}")


def _temporal_to_year_month(col: pa.Array) -> pa.Array:
    if pa.types.is_timestamp(col.type):
        years = pc.year(col).cast(pa.int64())
        months = pc.month(col).cast(pa.int64())
        return pc.add(pc.multiply(years, 100), months)
    if pa.types.is_integer(col.type):
        return col.cast(pa.int64())
    raise TypeError(f"Cannot extract year-month from {col.type}")


def _temporal_to_year_month_day(col: pa.Array) -> pa.Array:
    if pa.types.is_timestamp(col.type):
        years = pc.year(col).cast(pa.int64())
        months = pc.month(col).cast(pa.int64())
        days = pc.day(col).cast(pa.int64())
        return pc.add(pc.add(pc.multiply(years, 10000), pc.multiply(months, 100)), days)
    if pa.types.is_integer(col.type):
        return col.cast(pa.int64())
    raise TypeError(f"Cannot extract year-month-day from {col.type}")


def _scalar_to_year(value: object) -> int:
    if isinstance(value, int):
        return value
    import datetime

    if isinstance(value, datetime.datetime | datetime.date):
        return value.year
    import numpy as np

    if isinstance(value, np.datetime64):
        return int(np.datetime64(value, "Y").astype(int)) + 1970
    raise TypeError(f"Cannot extract year from {type(value).__name__}: {value!r}")


def _scalar_to_year_month(value: object) -> int:
    if isinstance(value, int):
        return value
    import datetime

    if isinstance(value, datetime.datetime | datetime.date):
        return value.year * 100 + value.month
    import numpy as np

    if isinstance(value, np.datetime64):
        dt = value.astype("datetime64[ms]").astype(datetime.datetime)
        return int(dt.year * 100 + dt.month)
    raise TypeError(f"Cannot extract year-month from {type(value).__name__}: {value!r}")


def _scalar_to_year_month_day(value: object) -> int:
    if isinstance(value, int):
        return value
    import datetime

    if isinstance(value, datetime.datetime | datetime.date):
        return value.year * 10000 + value.month * 100 + value.day
    import numpy as np

    if isinstance(value, np.datetime64):
        dt = value.astype("datetime64[ms]").astype(datetime.datetime)
        return int(dt.year * 10000 + dt.month * 100 + dt.day)
    raise TypeError(f"Cannot extract year-month-day from {type(value).__name__}: {value!r}")
