from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import ray.data

from rayzon.arrow import geodataframe_to_geoarrow_table
from rayzon.logging_utils import get_logger
from rayzon.types import COL_FEATURE_ID

if TYPE_CHECKING:
    import geopandas as gpd


logger = get_logger(__name__)


def to_feature_dataset(
    feature_source: gpd.GeoDataFrame | str | Path | ray.data.Dataset,
    *,
    override_num_blocks: int | None = None,
) -> ray.data.Dataset:
    if isinstance(feature_source, ray.data.Dataset):
        dataset = _maybe_override_num_blocks(feature_source, override_num_blocks)
        return _ensure_feature_id_column(dataset)
    if isinstance(feature_source, str | Path):
        dataset = ray.data.read_parquet(
            str(feature_source),
            include_paths=True,
            override_num_blocks=override_num_blocks,
        )
        return _ensure_feature_id_column(dataset)

    try:
        import geopandas as geopandas
    except ImportError as exc:
        raise ImportError(
            "GeoDataFrame support requires geopandas. "
            "Install with the 'geopandas' extra, for example "
            '`pip install "rayzon[geopandas]"` or '
            '`uv pip install "rayzon[geopandas]"`.'
        ) from exc

    if isinstance(feature_source, geopandas.GeoDataFrame):
        dataset = ray.data.from_arrow(geodataframe_to_geoarrow_table(feature_source))
        dataset = _maybe_override_num_blocks(dataset, override_num_blocks)
        return _ensure_feature_id_column(dataset)

    raise TypeError("feature_source must be a Ray Dataset, GeoDataFrame, or parquet path")


def _detect_feature_id_type(ds: ray.data.Dataset) -> pa.DataType:
    schema = ds.schema()
    if schema is None:
        return pa.string()

    arrow_schema = getattr(schema, "base_schema", None)
    if arrow_schema is not None and hasattr(arrow_schema, "field"):
        try:
            return arrow_schema.field(COL_FEATURE_ID).type
        except (KeyError, AttributeError):
            pass

    if hasattr(schema, "field"):
        try:
            return schema.field(COL_FEATURE_ID).type
        except (KeyError, AttributeError):
            pass
    return pa.string()


def _maybe_override_num_blocks(
    ds: ray.data.Dataset,
    override_num_blocks: int | None,
) -> ray.data.Dataset:
    if override_num_blocks is None:
        return ds
    if override_num_blocks <= 0:
        raise ValueError("override_num_blocks must be a positive integer when provided.")
    return ds.repartition(override_num_blocks)


def _detect_num_blocks(ds: ray.data.Dataset) -> int | None:
    try:
        return int(ds.num_blocks())
    except NotImplementedError:
        return None


def _ensure_feature_id_column(ds: ray.data.Dataset) -> ray.data.Dataset:
    schema = ds.schema()
    if schema is None or not hasattr(schema, "names"):
        return ds

    column_names = list(schema.names)
    if COL_FEATURE_ID in column_names:
        return ds

    logger.info(
        "feature_id missing; synthesizing integer %s by zipping with row range",
        COL_FEATURE_ID,
    )
    materialized = ds.materialize()
    row_index_ds = ray.data.range(
        materialized.count(),
        override_num_blocks=materialized.num_blocks(),
    ).rename_columns({"id": "__rayzon_feature_index"})
    result = materialized.zip(row_index_ds).rename_columns(
        {"__rayzon_feature_index": COL_FEATURE_ID}
    )
    return cast("ray.data.Dataset", result)
