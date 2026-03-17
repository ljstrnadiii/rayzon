from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
from shapely import from_wkb
from shapely.geometry.base import BaseGeometry

if TYPE_CHECKING:
    import geopandas as gpd


def as_table(batch: pa.Table | pa.RecordBatch) -> pa.Table:
    if isinstance(batch, pa.Table):
        return batch
    if isinstance(batch, pa.RecordBatch):
        return pa.Table.from_batches([batch])
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def coerce_geometry(value: BaseGeometry | bytes | bytearray | memoryview) -> BaseGeometry:
    if isinstance(value, BaseGeometry):
        return value
    return from_wkb(bytes(value))


def geodataframe_to_geoarrow_table(frame: gpd.GeoDataFrame) -> pa.Table:
    return pa.table(frame.to_arrow())
