from pathlib import Path

import geopandas as gpd
import pyarrow as pa
from shapely.geometry import Point

from rayzon.arrow import as_table, geodataframe_to_geoarrow_table
from rayzon.feature_dataset import _detect_feature_id_type, to_feature_dataset
from rayzon.types import COL_FEATURE_ID


def _gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            COL_FEATURE_ID: ["a", "b"],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def test_geodataframe_to_geoarrow_table_geometry_extension() -> None:
    table = geodataframe_to_geoarrow_table(_gdf())
    field = table.schema.field("geometry")
    metadata = field.metadata or {}

    assert metadata.get(b"ARROW:extension:name") == b"geoarrow.wkb"


def test_to_feature_dataset_from_geodataframe() -> None:
    dataset = to_feature_dataset(_gdf())
    assert dataset.count() == 2


def test_to_feature_dataset_geoarrow_extension_survives_in_ray_batch() -> None:
    dataset = to_feature_dataset(_gdf())
    first_batch = next(iter(dataset.iter_batches(batch_format="pyarrow", batch_size=10)))

    table = as_table(first_batch)
    field = table.schema.field("geometry")
    metadata = field.metadata or {}
    assert metadata.get(b"ARROW:extension:name") == b"geoarrow.wkb"


def test_to_feature_dataset_from_geoparquet_path(tmp_path: Path) -> None:
    path = tmp_path / "features.parquet"
    _gdf().to_parquet(path, index=False)

    dataset = to_feature_dataset(path)
    assert dataset.count() == 2

    first_batch = next(iter(dataset.iter_batches(batch_format="pyarrow", batch_size=10)))
    table = as_table(first_batch)
    field = table.schema.field("geometry")
    metadata = field.metadata or {}
    assert metadata.get(b"ARROW:extension:name") == b"geoarrow.wkb"


def test_to_feature_dataset_applies_override_num_blocks_for_geoparquet_path(tmp_path: Path) -> None:
    path = tmp_path / "features_blocks.parquet"
    _gdf().to_parquet(path, index=False)

    dataset = to_feature_dataset(path, override_num_blocks=2).materialize()

    assert dataset.num_blocks() == 2


def test_to_feature_dataset_applies_override_num_blocks_for_geodataframe() -> None:
    dataset = to_feature_dataset(_gdf(), override_num_blocks=2).materialize()

    assert dataset.num_blocks() == 2


def test_to_feature_dataset_synthesizes_integer_feature_id_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "features_without_feature_id.parquet"
    gdf = gpd.GeoDataFrame(
        {
            "value": [10, 20],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    gdf.to_parquet(path, index=False)

    dataset = to_feature_dataset(path)
    rows = dataset.take_all()

    assert len(rows) == 2
    assert COL_FEATURE_ID in rows[0]
    assert [row[COL_FEATURE_ID] for row in rows] == [0, 1]


def test_detect_feature_id_type_uses_arrow_base_schema_for_generated_ids(tmp_path: Path) -> None:
    path = tmp_path / "features_without_feature_id_for_type.parquet"
    gdf = gpd.GeoDataFrame(
        {
            "value": [10, 20],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    gdf.to_parquet(path, index=False)

    dataset = to_feature_dataset(path)

    assert _detect_feature_id_type(dataset) == pa.int64()
