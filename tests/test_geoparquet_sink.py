from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from shapely import Point, box, to_wkb

from rayzon.geoparquet_sink import (
    GeoParquetDatasink,
    _build_geo_metadata,
    _compute_bbox,
)

SAMPLE_CRS = {
    "type": "GeographicCRS",
    "name": "WGS 84",
    "id": {"authority": "EPSG", "code": 4326},
}


def _geometry_field(*, crs: dict | None = None, metadata: dict | None = None) -> pa.Field:
    if metadata is None and crs is not None:
        metadata = {
            b"ARROW:extension:name": b"geoarrow.wkb",
            b"ARROW:extension:metadata": json.dumps({"crs": crs}).encode(),
        }
    return pa.field("geometry", pa.binary(), metadata=metadata)


def _make_table(
    geoms: list[bytes | None],
    *,
    crs: dict | None = SAMPLE_CRS,
    extra_columns: dict | None = None,
) -> pa.Table:
    fields = [_geometry_field(crs=crs)]
    arrays: list[pa.Array] = [pa.array(geoms, type=pa.binary())]
    if extra_columns:
        for name, values in extra_columns.items():
            arrays.append(pa.array(values))
            fields.append(pa.field(name, arrays[-1].type))
    return pa.table(arrays, schema=pa.schema(fields))


def test_compute_bbox_single_table() -> None:
    wkb1 = to_wkb(Point(1.0, 2.0))
    wkb2 = to_wkb(Point(3.0, 4.0))
    table = _make_table([wkb1, wkb2])
    assert _compute_bbox(table, "geometry") == [1.0, 2.0, 3.0, 4.0]


def test_compute_bbox_multiple_points_via_concat() -> None:
    t1 = _make_table([to_wkb(box(0, 0, 1, 1))])
    t2 = _make_table([to_wkb(box(5, 5, 10, 10))])
    table = pa.concat_tables([t1, t2])
    assert _compute_bbox(table, "geometry") == [0.0, 0.0, 10.0, 10.0]


def test_compute_bbox_skips_none_geometries() -> None:
    table = _make_table([to_wkb(Point(1.0, 2.0)), None])
    assert _compute_bbox(table, "geometry") == [1.0, 2.0, 1.0, 2.0]


def test_compute_bbox_empty_table_returns_none() -> None:
    schema = pa.schema([_geometry_field(crs=SAMPLE_CRS)])
    table = pa.table({"geometry": pa.array([], type=pa.binary())}, schema=schema)
    assert _compute_bbox(table, "geometry") is None


def test_compute_bbox_no_geometry_column_returns_none() -> None:
    table = pa.table({"id": [1, 2]})
    assert _compute_bbox(table, "geometry") is None


def test_build_geo_metadata_injects_geo_key() -> None:
    table = _make_table([to_wkb(Point(1, 2))])
    meta = _build_geo_metadata(table, geometry_column="geometry", crs=SAMPLE_CRS)
    geo = json.loads(meta[b"geo"])
    assert geo["version"] == "1.1.0"
    assert geo["primary_column"] == "geometry"
    col = geo["columns"]["geometry"]
    assert col["encoding"] == "WKB"
    assert col["crs"] == SAMPLE_CRS
    assert col["bbox"] == [1.0, 2.0, 1.0, 2.0]


def test_build_geo_metadata_bbox_omitted_when_all_none() -> None:
    table = _make_table([None, None])
    meta = _build_geo_metadata(table, geometry_column="geometry", crs=SAMPLE_CRS)
    geo = json.loads(meta[b"geo"])
    assert "bbox" not in geo["columns"]["geometry"]


def test_geoparquet_datasink_write_with_explicit_crs(tmp_path: Path, ray_cluster: None) -> None:
    import ray.data

    # No extension metadata on geometry — simulates post-join data
    table = pa.table(
        {
            "geometry": pa.array([to_wkb(Point(i, i + 1)) for i in range(5)], type=pa.binary()),
            "value": list(range(5)),
        }
    )
    ds = ray.data.from_arrow(table)

    out_dir = str(tmp_path / "out")
    ds.write_datasink(GeoParquetDatasink(out_dir, crs=SAMPLE_CRS))

    result = pq.read_table(sorted((tmp_path / "out").glob("*.parquet"))[0])
    assert result.num_rows == 5
    geo = json.loads(result.schema.metadata[b"geo"])
    assert geo["version"] == "1.1.0"
    assert geo["columns"]["geometry"]["crs"] == SAMPLE_CRS


def test_geoparquet_datasink_extracts_crs_from_extension_metadata(
    tmp_path: Path, ray_cluster: None
) -> None:
    import ray.data

    table = _make_table(
        [to_wkb(Point(i, i + 1)) for i in range(3)],
        extra_columns={"value": list(range(3))},
    )
    ds = ray.data.from_arrow(table)

    out_dir = str(tmp_path / "out")
    ds.write_datasink(GeoParquetDatasink(out_dir))

    result = pq.read_table(sorted((tmp_path / "out").glob("*.parquet"))[0])
    geo = json.loads(result.schema.metadata[b"geo"])
    assert geo["columns"]["geometry"]["crs"] == SAMPLE_CRS
