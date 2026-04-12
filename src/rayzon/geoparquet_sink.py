from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.fs import FileSystem
from ray.data._internal.execution.interfaces import TaskContext
from ray.data.block import Block, BlockAccessor
from ray.data.datasource import Datasink

_GEOMETRY_COLUMN = "geometry"


def _compute_bbox(table: pa.Table, geometry_column: str) -> list[float] | None:
    from shapely import from_wkb

    if table.num_rows == 0 or geometry_column not in table.schema.names:
        return None
    bounds: list[tuple[float, float, float, float]] = []
    for wkb in table.column(geometry_column).to_pylist():
        if wkb is not None:
            bounds.append(from_wkb(bytes(wkb)).bounds)
    if not bounds:
        return None
    return [
        min(b[0] for b in bounds),
        min(b[1] for b in bounds),
        max(b[2] for b in bounds),
        max(b[3] for b in bounds),
    ]


def _build_geo_metadata(
    table: pa.Table,
    *,
    geometry_column: str,
    crs: dict | str,
) -> dict[bytes, bytes]:
    if isinstance(crs, str):
        crs_json: Any = crs
    else:
        crs_json = crs
    bbox = _compute_bbox(table, geometry_column)
    geo = {
        "version": "1.1.0",
        "primary_column": geometry_column,
        "columns": {
            geometry_column: {
                "encoding": "WKB",
                "geometry_types": [],
                "crs": crs_json,
                **({"bbox": bbox} if bbox else {}),
            }
        },
    }
    return {b"geo": json.dumps(geo).encode()}


class GeoParquetDatasink(Datasink):
    def __init__(
        self,
        path: str,
        *,
        crs: dict | str | None = None,
        geometry_column: str = _GEOMETRY_COLUMN,
        filesystem: FileSystem | None = None,
        parquet_write_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.path = path
        self.crs = crs
        self.geometry_column = geometry_column
        self.filesystem = filesystem
        self.parquet_write_args = parquet_write_args or {}

    def on_write_start(self, schema: pa.Schema | None = None) -> None:
        fs, path = (
            FileSystem.from_uri(self.path)
            if self.filesystem is None
            else (self.filesystem, self.path)
        )
        fs.create_dir(path, recursive=True)
        self._fs = fs
        self._base_path = path

    def write(self, blocks: Iterable[Block], ctx: TaskContext) -> None:
        tables = [
            BlockAccessor.for_block(b).to_arrow()
            for b in blocks
            if BlockAccessor.for_block(b).num_rows() > 0
        ]
        if not tables:
            return

        table = pa.concat_tables(tables)
        crs = self.crs or self._extract_crs(table.schema)
        if crs is None:
            metadata = {}
        else:
            metadata = _build_geo_metadata(table, geometry_column=self.geometry_column, crs=crs)

        schema = table.schema.with_metadata({**(table.schema.metadata or {}), **metadata})
        filename = f"part-{ctx.task_idx:05d}.parquet"
        path = f"{self._base_path}/{filename}"

        with self._fs.open_output_stream(path) as f:
            pq.write_table(table.cast(schema), f, **self.parquet_write_args)

    def _extract_crs(self, schema: pa.Schema) -> dict | None:
        if self.geometry_column not in schema.names:
            return None
        field = schema.field(self.geometry_column)
        if field.metadata is None:
            return None
        ext_meta_bytes = field.metadata.get(b"ARROW:extension:metadata")
        if ext_meta_bytes is None:
            return None
        return json.loads(ext_meta_bytes).get("crs")  # type: ignore[no-any-return]
