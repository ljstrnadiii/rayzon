from rzzs.reduce import reduce_partial_rows


def test_reduce_partial_rows_merges_values() -> None:
    partials = [
        {
            "feature_id": "p1",
            "dim_values": ("b1",),
            "count": 2,
            "n_valid": 2,
            "sum": 4.0,
            "sum_sq": 10.0,
            "min": 1.0,
            "max": 3.0,
        },
        {
            "feature_id": "p1",
            "dim_values": ("b1",),
            "count": 1,
            "n_valid": 1,
            "sum": 5.0,
            "sum_sq": 25.0,
            "min": 5.0,
            "max": 5.0,
        },
    ]
    out = reduce_partial_rows(partials)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["count"] == 3
    assert row["n_valid"] == 3
    assert row["sum"] == 9.0
    assert row["min"] == 1.0
    assert row["max"] == 5.0
