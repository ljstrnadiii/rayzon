from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def compute_partials(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    nodata: float | int | None,
    need_sum_sq: bool,
) -> dict[str, float | int]:
    selected = values[mask]
    count = int(selected.size)
    if count == 0:
        return {"count": 0, "n_valid": 0}

    if nodata is not None:
        selected = selected[selected != nodata]
    selected = selected[~np.isnan(selected)]
    n_valid = int(selected.size)
    if n_valid == 0:
        return {"count": count, "n_valid": 0}

    result: dict[str, float | int] = {
        "count": count,
        "n_valid": n_valid,
        "sum": float(selected.sum()),
        "min": float(selected.min()),
        "max": float(selected.max()),
    }
    if need_sum_sq:
        result["sum_sq"] = float(np.square(selected).sum())
    return result


def finalize_moments(n_valid: int, sum_: float, sum_sq: float | None) -> dict[str, float | int]:
    if n_valid == 0:
        return {
            "n_valid": 0,
            "sum": 0.0,
            "mean": math.nan,
            "variance": math.nan,
            "std": math.nan,
        }

    mean = sum_ / n_valid
    variance = math.nan
    std = math.nan
    if sum_sq is not None:
        variance = max(0.0, (sum_sq / n_valid) - (mean * mean))
        std = math.sqrt(variance)

    return {
        "n_valid": n_valid,
        "sum": sum_,
        "mean": mean,
        "variance": variance,
        "std": std,
    }


def merge_min(values: Iterable[float]) -> float:
    return min(values)


def merge_max(values: Iterable[float]) -> float:
    return max(values)
