from __future__ import annotations

import datetime

import numpy as np
import pyarrow as pa
import pytest
import ray.data

from rayzon.coord_align import (
    add_coord_column,
    build_coord_to_chunk_ids,
    coord_col_name,
    is_coord_col,
    match_coord_values_to_dim_indices,
    normalize_coord_columns,
    truncate_arrow_column,
    truncate_coord_value,
)


class TestNormalize:
    def test_none(self) -> None:
        assert normalize_coord_columns(None) == {}

    def test_sequence(self) -> None:
        assert normalize_coord_columns(("time",)) == {"time": None}

    def test_mapping(self) -> None:
        assert normalize_coord_columns({"time": "Y"}) == {"time": "Y"}

    def test_invalid_frequency(self) -> None:
        with pytest.raises(ValueError, match="Invalid frequency"):
            normalize_coord_columns({"time": "Q"})


class TestCoordColName:
    def test_name(self) -> None:
        assert coord_col_name("time") == "__time_coord"

    def test_is_coord_col(self) -> None:
        assert is_coord_col("__time_coord")
        assert not is_coord_col("time")
        assert not is_coord_col("__time")


class TestTruncateArrowColumn:
    def test_none_passthrough_int(self) -> None:
        col = pa.array([2024, 2025], type=pa.int64())
        result = truncate_arrow_column(col, None)
        assert result.to_pylist() == [2024, 2025]

    def test_none_timestamp_to_int(self) -> None:
        col = pa.array(
            [datetime.datetime(2024, 1, 1), datetime.datetime(2025, 6, 15)],
            type=pa.timestamp("us"),
        )
        result = truncate_arrow_column(col, None)
        assert pa.types.is_integer(result.type)

    def test_year_from_timestamp(self) -> None:
        col = pa.array(
            [datetime.datetime(2024, 3, 15), datetime.datetime(2024, 9, 1)],
            type=pa.timestamp("us"),
        )
        result = truncate_arrow_column(col, "Y")
        assert result.to_pylist() == [2024, 2024]

    def test_year_from_int(self) -> None:
        col = pa.array([2024, 2025], type=pa.int64())
        result = truncate_arrow_column(col, "Y")
        assert result.to_pylist() == [2024, 2025]

    def test_month_from_timestamp(self) -> None:
        col = pa.array(
            [datetime.datetime(2024, 1, 15), datetime.datetime(2024, 12, 1)],
            type=pa.timestamp("us"),
        )
        result = truncate_arrow_column(col, "M")
        assert result.to_pylist() == [202401, 202412]

    def test_day_from_timestamp(self) -> None:
        col = pa.array(
            [datetime.datetime(2024, 1, 15), datetime.datetime(2024, 12, 31)],
            type=pa.timestamp("us"),
        )
        result = truncate_arrow_column(col, "D")
        assert result.to_pylist() == [20240115, 20241231]


class TestTruncateCoordValue:
    def test_none_passthrough(self) -> None:
        assert truncate_coord_value(2024, None) == 2024

    def test_year_from_int(self) -> None:
        assert truncate_coord_value(2024, "Y") == 2024

    def test_year_from_datetime(self) -> None:
        assert truncate_coord_value(datetime.datetime(2024, 6, 15), "Y") == 2024

    def test_year_from_date(self) -> None:
        assert truncate_coord_value(datetime.date(2024, 6, 15), "Y") == 2024

    def test_year_from_np_datetime64(self) -> None:
        assert truncate_coord_value(np.datetime64("2024-06-15"), "Y") == 2024

    def test_month_from_datetime(self) -> None:
        assert truncate_coord_value(datetime.datetime(2024, 6, 15), "M") == 202406

    def test_day_from_datetime(self) -> None:
        assert truncate_coord_value(datetime.datetime(2024, 6, 15), "D") == 20240615


class TestMatchCoordValues:
    def test_exact_int_match(self) -> None:
        result = match_coord_values_to_dim_indices([2024], [2023, 2024, 2025], None)
        assert result == {2024: [1]}

    def test_year_match_multiple_indices(self) -> None:
        # zarr has monthly coords as datetimes, feature extracts year
        zarr_coords = [
            datetime.datetime(2024, 1, 1),
            datetime.datetime(2024, 6, 1),
            datetime.datetime(2025, 1, 1),
        ]
        result = match_coord_values_to_dim_indices([2024], zarr_coords, "Y")
        assert result == {2024: [0, 1]}

    def test_no_match_returns_empty(self) -> None:
        result = match_coord_values_to_dim_indices([2030], [2023, 2024], None)
        assert result == {}


class TestBuildCoordToChunkIds:
    def test_single_chunk(self) -> None:
        result = build_coord_to_chunk_ids({2024: [0, 1, 2]}, chunk_size=10)
        assert result == {2024: [0]}

    def test_multiple_chunks(self) -> None:
        result = build_coord_to_chunk_ids({2024: [5, 15]}, chunk_size=10)
        assert result == {2024: [0, 1]}


class TestAddCoordColumn:
    def test_adds_derived_column(self) -> None:
        ds = ray.data.from_arrow(
            pa.table(
                {
                    "feature_id": ["a", "b"],
                    "time": pa.array(
                        [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 6, 15)],
                        type=pa.timestamp("us"),
                    ),
                }
            )
        )
        result = add_coord_column(ds, "time", "Y")
        rows = result.take_all()
        assert "__time_coord" in rows[0]
        assert rows[0]["__time_coord"] == 2024
        assert rows[1]["__time_coord"] == 2024
        # Original column preserved
        assert "time" in rows[0]
