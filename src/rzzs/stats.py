from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from ray.data.aggregate import (
    AggregateFnV2,
    BlockAccessor,
)
from ray.data.block import Block
from rayzon.types import COL_PIXELS

DEFAULT_STAT_EXPRS: tuple[str, ...] = ("count", "n_valid", "mean", "std")

_QUANTILE_RE = re.compile(r"^p_?(100|[0-9]{1,2})$")

_BUILTIN_NAMES: frozenset[str] = frozenset(
    {"count", "n_valid", "sum", "mean", "variance", "std", "min", "max"}
)


def normalize_stat_expr(expr: str) -> str:
    normalized = expr.strip().lower().replace(" ", "")
    if normalized in _BUILTIN_NAMES:
        return normalized
    match = _QUANTILE_RE.fullmatch(normalized)
    if match is not None:
        return f"p{int(match.group(1))}"
    raise ValueError(
        f"Unsupported stat expression '{expr}'. "
        f"Use builtins ({sorted(_BUILTIN_NAMES)}) or quantiles like 'p50'/'p_73'."
    )


@dataclass(frozen=True)
class ResolvedStats:
    requested: tuple[str, ...]
    quantiles: tuple[float, ...]


def resolve_stat_exprs(stats: tuple[str, ...] | list[str]) -> ResolvedStats:
    normalized = tuple(dict.fromkeys(normalize_stat_expr(e) for e in stats))

    quantiles: list[float] = []
    for name in normalized:
        if name.startswith("p") and name not in _BUILTIN_NAMES:
            quantiles.append(int(name[1:]) / 100.0)

    return ResolvedStats(requested=normalized, quantiles=tuple(quantiles))


def _flatten_pixels(block: Block) -> np.ndarray:
    table = BlockAccessor.for_block(block).to_arrow()
    if table.num_rows == 0:
        return np.array([], dtype=np.float64)
    col = table.column(COL_PIXELS).combine_chunks()
    flat = col.values
    if len(flat) == 0:
        return np.array([], dtype=np.float64)
    return flat.to_numpy(zero_copy_only=False)  # type: ignore[no-any-return]


class CountAgg(AggregateFnV2[int | None, int]):
    def __init__(self, *, alias_name: str = "count") -> None:
        super().__init__(
            alias_name,
            on=COL_PIXELS,
            ignore_nulls=True,
            zero_factory=lambda: 0,
        )

    def aggregate_block(self, block: Block) -> int | None:
        flat = _flatten_pixels(block)
        return len(flat) if len(flat) > 0 else None

    def combine(
        self,
        current_accumulator: int | None,
        new: int | None,
    ) -> int | None:
        if new is None:
            return current_accumulator
        if current_accumulator is None:
            return new
        return current_accumulator + new

    def finalize(self, accumulator: int | None) -> int | None:  # type: ignore[override]
        return accumulator if accumulator is not None else 0


class NValidAgg(AggregateFnV2[int | None, int]):
    def __init__(self, *, alias_name: str = "n_valid") -> None:
        super().__init__(
            alias_name,
            on=COL_PIXELS,
            ignore_nulls=True,
            zero_factory=lambda: 0,
        )

    def aggregate_block(self, block: Block) -> int | None:
        flat = _flatten_pixels(block)
        if len(flat) == 0:
            return None
        valid = flat[~np.isnan(flat)]
        return len(valid) if len(valid) > 0 else None

    def combine(
        self,
        current_accumulator: int | None,
        new: int | None,
    ) -> int | None:
        if new is None:
            return current_accumulator
        if current_accumulator is None:
            return new
        return current_accumulator + new

    def finalize(self, accumulator: int | None) -> int | None:  # type: ignore[override]
        return accumulator if accumulator is not None else 0


class SumAgg(AggregateFnV2[float | None, float]):
    def __init__(self, *, alias_name: str = "sum") -> None:
        super().__init__(
            alias_name,
            on=COL_PIXELS,
            ignore_nulls=True,
            zero_factory=lambda: 0.0,
        )

    def aggregate_block(self, block: Block) -> float | None:
        flat = _flatten_pixels(block)
        valid = flat[~np.isnan(flat)]
        if len(valid) == 0:
            return None
        return float(valid.sum())

    def combine(
        self,
        current_accumulator: float | None,
        new: float | None,
    ) -> float | None:
        if new is None:
            return current_accumulator
        if current_accumulator is None:
            return new
        return current_accumulator + new

    def finalize(self, accumulator: float | None) -> float | None:  # type: ignore[override]
        return accumulator if accumulator is not None else 0.0


class MeanAgg(AggregateFnV2[list[int | float] | None, float]):
    def __init__(self, *, alias_name: str = "mean") -> None:
        super().__init__(
            alias_name,
            on=COL_PIXELS,
            ignore_nulls=True,
            zero_factory=lambda: [0.0, 0],
        )

    def aggregate_block(self, block: Block) -> list[int | float] | None:
        flat = _flatten_pixels(block)
        valid = flat[~np.isnan(flat)]
        n = len(valid)
        if n == 0:
            return None
        return [float(valid.sum()), n]

    def combine(
        self,
        current_accumulator: list[int | float] | None,
        new: list[int | float] | None,
    ) -> list[int | float] | None:
        if new is None:
            return current_accumulator
        if current_accumulator is None:
            return new
        return [
            current_accumulator[0] + new[0],
            current_accumulator[1] + new[1],
        ]

    def finalize(  # type: ignore[override]
        self,
        accumulator: list[int | float] | None,
    ) -> float | None:
        if accumulator is None:
            return float("nan")
        s, n = accumulator
        if n == 0:
            return float("nan")
        return s / n


class VarianceAgg(AggregateFnV2[list[int | float] | None, float]):
    def __init__(self, *, alias_name: str = "variance") -> None:
        super().__init__(
            alias_name,
            on=COL_PIXELS,
            ignore_nulls=True,
            zero_factory=lambda: [0.0, 0.0, 0],
        )

    def aggregate_block(self, block: Block) -> list[int | float] | None:
        flat = _flatten_pixels(block)
        valid = flat[~np.isnan(flat)]
        n = len(valid)
        if n == 0:
            return None
        return [float(np.square(valid).sum()), float(valid.sum()), n]

    def combine(
        self,
        current_accumulator: list[int | float] | None,
        new: list[int | float] | None,
    ) -> list[int | float] | None:
        if new is None:
            return current_accumulator
        if current_accumulator is None:
            return new
        return [
            current_accumulator[0] + new[0],
            current_accumulator[1] + new[1],
            current_accumulator[2] + new[2],
        ]

    def finalize(  # type: ignore[override]
        self,
        accumulator: list[int | float] | None,
    ) -> float | None:
        if accumulator is None:
            return float("nan")
        sq, s, n = accumulator
        if n == 0:
            return float("nan")
        mean = s / n
        return max(0.0, sq / n - mean * mean)


class StdAgg(AggregateFnV2[list[int | float] | None, float]):
    def __init__(self, *, alias_name: str = "std") -> None:
        super().__init__(
            alias_name,
            on=COL_PIXELS,
            ignore_nulls=True,
            zero_factory=lambda: [0.0, 0.0, 0],
        )

    def aggregate_block(self, block: Block) -> list[int | float] | None:
        flat = _flatten_pixels(block)
        valid = flat[~np.isnan(flat)]
        n = len(valid)
        if n == 0:
            return None
        return [float(np.square(valid).sum()), float(valid.sum()), n]

    def combine(
        self,
        current_accumulator: list[int | float] | None,
        new: list[int | float] | None,
    ) -> list[int | float] | None:
        if new is None:
            return current_accumulator
        if current_accumulator is None:
            return new
        return [
            current_accumulator[0] + new[0],
            current_accumulator[1] + new[1],
            current_accumulator[2] + new[2],
        ]

    def finalize(  # type: ignore[override]
        self,
        accumulator: list[int | float] | None,
    ) -> float | None:
        if accumulator is None:
            return float("nan")
        sq, s, n = accumulator
        if n == 0:
            return float("nan")
        mean = s / n
        return math.sqrt(max(0.0, sq / n - mean * mean))


class MinAgg(AggregateFnV2[float | None, float]):
    def __init__(self, *, alias_name: str = "min") -> None:
        super().__init__(
            alias_name,
            on=COL_PIXELS,
            ignore_nulls=True,
            zero_factory=lambda: float("inf"),
        )

    def aggregate_block(self, block: Block) -> float | None:
        flat = _flatten_pixels(block)
        valid = flat[~np.isnan(flat)]
        if len(valid) == 0:
            return None
        return float(valid.min())

    def combine(
        self,
        current_accumulator: float | None,
        new: float | None,
    ) -> float | None:
        if new is None:
            return current_accumulator
        if current_accumulator is None:
            return new
        return min(current_accumulator, new)

    def finalize(self, accumulator: float | None) -> float | None:  # type: ignore[override]
        if accumulator is None or accumulator == float("inf"):
            return float("nan")
        return accumulator


class MaxAgg(AggregateFnV2[float | None, float]):
    def __init__(self, *, alias_name: str = "max") -> None:
        super().__init__(
            alias_name,
            on=COL_PIXELS,
            ignore_nulls=True,
            zero_factory=lambda: float("-inf"),
        )

    def aggregate_block(self, block: Block) -> float | None:
        flat = _flatten_pixels(block)
        valid = flat[~np.isnan(flat)]
        if len(valid) == 0:
            return None
        return float(valid.max())

    def combine(
        self,
        current_accumulator: float | None,
        new: float | None,
    ) -> float | None:
        if new is None:
            return current_accumulator
        if current_accumulator is None:
            return new
        return max(current_accumulator, new)

    def finalize(self, accumulator: float | None) -> float | None:  # type: ignore[override]
        if accumulator is None or accumulator == float("-inf"):
            return float("nan")
        return accumulator


_AGG_BUILDERS: dict[str, Callable[[], AggregateFnV2]] = {
    "count": lambda: CountAgg(),
    "n_valid": lambda: NValidAgg(),
    "sum": lambda: SumAgg(),
    "mean": lambda: MeanAgg(),
    "variance": lambda: VarianceAgg(),
    "std": lambda: StdAgg(),
    "min": lambda: MinAgg(),
    "max": lambda: MaxAgg(),
}


def build_aggregations(resolved: ResolvedStats) -> list[AggregateFnV2]:
    aggs: list[AggregateFnV2] = []
    for name in resolved.requested:
        builder = _AGG_BUILDERS.get(name)
        if builder is not None:
            aggs.append(builder())
    return aggs
