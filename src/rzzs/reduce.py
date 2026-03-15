from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import pandas as pd

from rzzs.partials import finalize_moments


def reduce_partial_rows(partials: Iterable[dict], *, include_variance: bool = True) -> pd.DataFrame:
    grouped: dict[tuple, dict[str, float | int]] = defaultdict(
        lambda: {
            "count": 0,
            "n_valid": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": float("inf"),
            "max": float("-inf"),
        }
    )

    for row in partials:
        key = (row["feature_id"], row.get("dim_values", ()))
        agg = grouped[key]
        count = int(row.get("count", 0))
        n_valid = int(row.get("n_valid", 0))
        if count == 0:
            continue

        agg["count"] += count
        agg["n_valid"] += n_valid
        agg["sum"] += float(row.get("sum", 0.0))
        agg["sum_sq"] += float(row.get("sum_sq", 0.0))
        if n_valid > 0:
            agg["min"] = min(float(agg["min"]), float(row.get("min", float("inf"))))
            agg["max"] = max(float(agg["max"]), float(row.get("max", float("-inf"))))

    out_rows: list[dict] = []
    for (feature_id, dim_values), agg in grouped.items():
        sum_sq = float(agg["sum_sq"]) if include_variance else None
        moments = finalize_moments(int(agg["n_valid"]), float(agg["sum"]), sum_sq)
        min_value = float(agg["min"]) if int(agg["n_valid"]) > 0 else float("nan")
        max_value = float(agg["max"]) if int(agg["n_valid"]) > 0 else float("nan")
        out_rows.append(
            {
                "feature_id": feature_id,
                "dim_values": dim_values,
                "count": int(agg["count"]),
                **moments,
                "min": min_value,
                "max": max_value,
            }
        )

    return pd.DataFrame(out_rows)
