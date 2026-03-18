from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
import pyarrow as pa
from ray.data.aggregate import AggregateFnV2, BlockAccessor
from ray.data.block import Block
from tdigest import TDigest

DEFAULT_STAT_EXPRS: tuple[str, ...] = ("count", "n_valid", "mean", "std")

_QUANTILE_RE = re.compile(r"^p_?(100|[0-9]{1,2})$")

TDIGEST_DEFAULT_DELTA = 0.001
TDIGEST_DEFAULT_K = 200

PartialValue: TypeAlias = int | float | np.ndarray | list[float] | None
PartialState: TypeAlias = dict[str, PartialValue]


@dataclass(frozen=True)
class ResolvedStats:
    requested: tuple[str, ...]
    quantiles: tuple[float, ...]


class Stat:
    stat_name: str | None = None
    name: str | None = None
    arrow_type: pa.DataType | None = None
    default: PartialValue = None
    required_partial_columns: tuple[type[Stat], ...] = ()

    @property
    def is_partial_field(self) -> bool:
        return self.name is not None and self.arrow_type is not None

    def accumulate(self, values: np.ndarray) -> PartialValue:
        raise NotImplementedError

    def merge(self, left: PartialValue, right: PartialValue) -> PartialValue:
        raise NotImplementedError

    def finalize_output(self, state: PartialState | None) -> int | float | None:
        raise NotImplementedError


class CountStat(Stat):
    stat_name = "count"
    name = "_partial_count"
    arrow_type = pa.int64()
    default: PartialValue = 0

    def accumulate(self, values: np.ndarray) -> PartialValue:
        return int(values.size)

    def merge(self, left: PartialValue, right: PartialValue) -> PartialValue:
        return _as_int(left) + _as_int(right)

    def finalize_output(self, state: PartialState | None) -> int:
        return _as_int(_value_for_column(state, self.name, self.default))


class NValidStat(Stat):
    stat_name = "n_valid"
    name = "_partial_n_valid"
    arrow_type = pa.int64()
    default: PartialValue = 0

    def accumulate(self, values: np.ndarray) -> PartialValue:
        return _valid_count(values)

    def merge(self, left: PartialValue, right: PartialValue) -> PartialValue:
        return _as_int(left) + _as_int(right)

    def finalize_output(self, state: PartialState | None) -> int:
        return _as_int(_value_for_column(state, self.name, self.default))


class SumStat(Stat):
    stat_name = "sum"
    name = "_partial_sum"
    arrow_type = pa.float64()
    default: PartialValue = 0.0

    def accumulate(self, values: np.ndarray) -> PartialValue:
        return _valid_sum(values)

    def merge(self, left: PartialValue, right: PartialValue) -> PartialValue:
        return _as_float(left) + _as_float(right)

    def finalize_output(self, state: PartialState | None) -> float:
        return _as_float(_value_for_column(state, self.name, self.default))


class SumSqStat(Stat):
    name = "_partial_sum_sq"
    arrow_type = pa.float64()
    default: PartialValue = 0.0

    def accumulate(self, values: np.ndarray) -> PartialValue:
        return _valid_sum_sq(values)

    def merge(self, left: PartialValue, right: PartialValue) -> PartialValue:
        return _as_float(left) + _as_float(right)


class MinStat(Stat):
    stat_name = "min"
    name = "_partial_min"
    arrow_type = pa.float64()
    default: PartialValue = float("inf")

    def accumulate(self, values: np.ndarray) -> PartialValue:
        return _valid_min(values)

    def merge(self, left: PartialValue, right: PartialValue) -> PartialValue:
        return min(_as_float(left), _as_float(right))

    def finalize_output(self, state: PartialState | None) -> float:
        min_value = _as_float(_value_for_column(state, self.name, self.default))
        return float("nan") if math.isinf(min_value) else min_value


class MaxStat(Stat):
    stat_name = "max"
    name = "_partial_max"
    arrow_type = pa.float64()
    default: PartialValue = float("-inf")

    def accumulate(self, values: np.ndarray) -> PartialValue:
        return _valid_max(values)

    def merge(self, left: PartialValue, right: PartialValue) -> PartialValue:
        return max(_as_float(left), _as_float(right))

    def finalize_output(self, state: PartialState | None) -> float:
        max_value = _as_float(_value_for_column(state, self.name, self.default))
        return float("nan") if math.isinf(max_value) else max_value


class DigestStat(Stat):
    name = "_partial_digest"
    arrow_type = pa.list_(pa.float64())
    default: PartialValue = None

    def accumulate(self, values: np.ndarray) -> PartialValue:
        valid = _valid_values(values)
        return _build_digest_payload(valid) if valid.size > 0 else None

    def merge(self, left: PartialValue, right: PartialValue) -> PartialValue:
        return _merge_digest_payloads(
            _coerce_digest_payload(left),
            _coerce_digest_payload(right),
        )


class MeanStat(Stat):
    stat_name = "mean"
    required_partial_columns = (NValidStat, SumStat)

    def finalize_output(self, state: PartialState | None) -> float:
        count = _as_int(_value_for_column(state, NValidStat.name, 0))
        value_sum = _as_float(_value_for_column(state, SumStat.name, 0.0))
        if count == 0:
            return float("nan")
        return value_sum / count


class VarianceStat(Stat):
    stat_name = "variance"
    required_partial_columns = (NValidStat, SumStat, SumSqStat)

    def finalize_output(self, state: PartialState | None) -> float:
        n_valid = _as_int(_value_for_column(state, NValidStat.name, 0))
        if n_valid == 0:
            return float("nan")

        value_sum = _as_float(_value_for_column(state, SumStat.name, 0.0))
        value_sum_sq = _as_float(_value_for_column(state, SumSqStat.name, 0.0))
        mean = value_sum / n_valid
        return max(0.0, value_sum_sq / n_valid - mean * mean)


class StdStat(Stat):
    stat_name = "std"
    required_partial_columns = (NValidStat, SumStat, SumSqStat)

    def finalize_output(self, state: PartialState | None) -> float:
        n_valid = _as_int(_value_for_column(state, NValidStat.name, 0))
        if n_valid == 0:
            return float("nan")

        value_sum = _as_float(_value_for_column(state, SumStat.name, 0.0))
        value_sum_sq = _as_float(_value_for_column(state, SumSqStat.name, 0.0))
        mean = value_sum / n_valid
        variance = max(0.0, value_sum_sq / n_valid - mean * mean)
        return math.sqrt(variance)


class QuantileStat(Stat):
    required_partial_columns = (DigestStat,)

    def __init__(self, *, quantile: float) -> None:
        self._quantile = quantile
        self.stat_name = f"p{int(quantile * 100)}"

    def finalize_output(self, state: PartialState | None) -> float:
        digest_payload = _coerce_digest_payload(_value_for_column(state, DigestStat.name, None))
        if digest_payload is None:
            return float("nan")

        digest = _deserialize_digest(digest_payload)
        return float(digest.percentile(self._quantile * 100.0))


def _valid_values(values: np.ndarray) -> np.ndarray:
    return values[~np.isnan(values)]


def _valid_sum(values: np.ndarray) -> float:
    valid = _valid_values(values)
    return float(valid.sum()) if valid.size > 0 else 0.0


def _valid_count(values: np.ndarray) -> int:
    return int(_valid_values(values).size)


def _valid_sum_sq(values: np.ndarray) -> float:
    valid = _valid_values(values)
    return float(np.square(valid).sum()) if valid.size > 0 else 0.0


def _valid_min(values: np.ndarray) -> float:
    valid = _valid_values(values)
    return float(valid.min()) if valid.size > 0 else float("inf")


def _valid_max(values: np.ndarray) -> float:
    valid = _valid_values(values)
    return float(valid.max()) if valid.size > 0 else float("-inf")


def _as_int(value: PartialValue) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value)
    raise TypeError(f"Expected integer-like partial value, got {type(value)!r}")


def _as_float(value: PartialValue) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    raise TypeError(f"Expected numeric partial value, got {type(value)!r}")


def _value_for_column(
    state: PartialState | None,
    column: str,
    default: PartialValue,
) -> PartialValue:
    if state is None:
        return default
    return state.get(column, default)


COUNT_STAT = CountStat()
N_VALID_STAT = NValidStat()
SUM_STAT = SumStat()
SUM_SQ_STAT = SumSqStat()
MIN_STAT = MinStat()
MAX_STAT = MaxStat()
DIGEST_STAT = DigestStat()
MEAN_STAT = MeanStat()
VARIANCE_STAT = VarianceStat()
STD_STAT = StdStat()

PARTIAL_STAT_REGISTRY_BY_CLASS: dict[type[Stat], Stat] = {
    CountStat: COUNT_STAT,
    NValidStat: N_VALID_STAT,
    SumStat: SUM_STAT,
    SumSqStat: SUM_SQ_STAT,
    MinStat: MIN_STAT,
    MaxStat: MAX_STAT,
    DigestStat: DIGEST_STAT,
}

_STAT_REGISTRY: dict[str, Stat] = {
    stat.stat_name: stat
    for stat in (
        COUNT_STAT,
        N_VALID_STAT,
        SUM_STAT,
        MEAN_STAT,
        VARIANCE_STAT,
        STD_STAT,
        MIN_STAT,
        MAX_STAT,
    )
    if stat.stat_name is not None
}

PARTIAL_FIELD_BY_NAME: dict[str, Stat] = {
    stat.name: stat for stat in PARTIAL_STAT_REGISTRY_BY_CLASS.values() if stat.name is not None
}

ALL_PARTIAL_COLUMNS: tuple[str, ...] = tuple(PARTIAL_FIELD_BY_NAME.keys())


def normalize_stat_expr(expr: str) -> str:
    normalized = expr.strip().lower().replace(" ", "")
    if normalized in _STAT_REGISTRY:
        return normalized
    match = _QUANTILE_RE.fullmatch(normalized)
    if match is not None:
        return f"p{int(match.group(1))}"
    raise ValueError(
        f"Unsupported stat expression '{expr}'. "
        f"Use builtins ({sorted(_STAT_REGISTRY)}) or quantiles like 'p50'/'p_73'."
    )


def resolve_stat_exprs(stats: tuple[str, ...] | list[str]) -> ResolvedStats:
    normalized = tuple(dict.fromkeys(normalize_stat_expr(e) for e in stats))

    quantiles: list[float] = []
    for name in normalized:
        match = _QUANTILE_RE.fullmatch(name)
        if match is not None:
            quantiles.append(int(match.group(1)) / 100.0)

    return ResolvedStats(requested=normalized, quantiles=tuple(quantiles))


def _serialize_digest(digest: TDigest) -> np.ndarray:
    centroids = digest.centroids_to_list()
    if not centroids:
        return np.array([], dtype=np.float64)

    payload = np.empty(2 * len(centroids), dtype=np.float64)
    for idx, centroid in enumerate(centroids):
        payload[2 * idx] = float(centroid["m"])
        payload[2 * idx + 1] = float(centroid["c"])
    return payload


def _build_digest_payload(valid_values: np.ndarray) -> np.ndarray:
    digest = TDigest(delta=TDIGEST_DEFAULT_DELTA, K=TDIGEST_DEFAULT_K)
    digest.batch_update(valid_values.tolist())
    digest.compress()
    return _serialize_digest(digest)


def _coerce_digest_payload(payload: PartialValue) -> np.ndarray | None:
    if payload is None:
        return None
    if isinstance(payload, np.ndarray):
        return payload.astype(np.float64, copy=False)
    if isinstance(payload, list):
        return np.asarray(payload, dtype=np.float64)
    raise TypeError(f"Unsupported digest payload type: {type(payload)!r}")


def _deserialize_digest(payload: np.ndarray) -> TDigest:
    digest = TDigest(delta=TDIGEST_DEFAULT_DELTA, K=TDIGEST_DEFAULT_K)
    flat = np.asarray(payload, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return digest
    if flat.size % 2 != 0:
        raise ValueError("Malformed digest payload; expected even-length centroid array.")

    centroids = [{"m": float(flat[i]), "c": float(flat[i + 1])} for i in range(0, flat.size, 2)]
    digest.update_centroids_from_list(centroids)
    digest.compress()
    return digest


def _merge_digest_payloads(current: np.ndarray | None, new: np.ndarray | None) -> np.ndarray | None:
    if new is None:
        return current
    if current is None:
        return new

    current_digest = _deserialize_digest(current)
    merged = current_digest + _deserialize_digest(new)
    merged.compress()
    return _serialize_digest(merged)


def partial_fields_for_columns(columns: tuple[str, ...] | list[str]) -> tuple[Stat, ...]:
    out: list[Stat] = []
    seen: set[str] = set()
    for column in columns:
        field = PARTIAL_FIELD_BY_NAME.get(column)
        if field is None:
            raise ValueError(f"Unsupported partial-state column: {column}")
        if field.name is None:
            continue
        if field.name not in seen:
            seen.add(field.name)
            out.append(field)
    return tuple(out)


def partial_column_arrow_type(column: str) -> pa.DataType:
    field = PARTIAL_FIELD_BY_NAME.get(column)
    if field is None or field.arrow_type is None:
        raise ValueError(f"Unsupported partial-state column: {column}")
    return field.arrow_type


def accumulate_partial_state(
    values: np.ndarray,
    *,
    required_fields: tuple[Stat, ...],
) -> PartialState | None:
    if not required_fields:
        return None

    flat_values = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat_values.size == 0:
        return None

    state: PartialState = {}
    for field in required_fields:
        if field.name is None:
            continue
        state[field.name] = field.accumulate(flat_values)
    return state


def merge_partial_states(
    current: PartialState | None,
    new: PartialState | None,
    *,
    include_digest: bool,
) -> PartialState | None:
    if new is None:
        return current
    if current is None:
        if include_digest:
            return dict(new)
        return {k: v for k, v in new.items() if k != DigestStat.name}

    merged = dict(current)
    for column, value in new.items():
        if column == DigestStat.name and not include_digest:
            continue

        field = PARTIAL_FIELD_BY_NAME.get(column)
        if field is None:
            raise ValueError(f"Unsupported partial-state column: {column}")
        previous = merged.get(column, field.default)
        merged[column] = field.merge(previous, value)
    return merged


def _stat_partial_fields(stat: Stat) -> tuple[Stat, ...]:
    fields: list[Stat] = []
    seen: set[str] = set()

    def add_field(field: Stat) -> None:
        for dependency_class in field.required_partial_columns:
            dependency = PARTIAL_STAT_REGISTRY_BY_CLASS.get(dependency_class)
            if dependency is None:
                raise ValueError(
                    f"Stat '{field.stat_name or field.name}' depends on unknown partial stat "
                    f"'{dependency_class.__name__}'."
                )
            add_field(dependency)
        if field.name is not None and field.name not in seen:
            seen.add(field.name)
            fields.append(field)

    if stat.is_partial_field:
        add_field(stat)

    for dependency_class in stat.required_partial_columns:
        dependency = PARTIAL_STAT_REGISTRY_BY_CLASS.get(dependency_class)
        if dependency is None:
            raise ValueError(
                f"Stat '{stat.stat_name or stat.name}' depends on unknown partial stat "
                f"'{dependency_class.__name__}'."
            )
        add_field(dependency)

    return tuple(fields)


def _stat_for_name(name: str) -> Stat | None:
    builtin = _STAT_REGISTRY.get(name)
    if builtin is not None:
        return builtin

    match = _QUANTILE_RE.fullmatch(name)
    if match is not None:
        quantile_int = int(match.group(1))
        return QuantileStat(quantile=quantile_int / 100.0)

    return None


class PartialStateAgg(AggregateFnV2[PartialState | None, int | float | None]):
    def __init__(self, stat: Stat) -> None:
        if stat.stat_name is None:
            raise ValueError("Aggregation stat must define stat_name.")
        self._stat = stat
        self._partial_fields = _stat_partial_fields(stat)
        self._include_digest = any(field.name == DigestStat.name for field in self._partial_fields)
        super().__init__(
            stat.stat_name,
            on=None,
            ignore_nulls=True,
            zero_factory=lambda: None,
        )

    def aggregate_block(self, block: Block) -> PartialState | None:
        table = BlockAccessor.for_block(block).to_arrow()
        if table.num_rows == 0:
            return None

        columns: dict[str, list[PartialValue]] = {}
        for field in self._partial_fields:
            if field.name is None:
                continue
            if field.name in table.column_names:
                columns[field.name] = cast(list[PartialValue], table.column(field.name).to_pylist())
            else:
                columns[field.name] = [field.default] * table.num_rows

        accumulator: PartialState | None = None
        for row_idx in range(table.num_rows):
            row_partial: PartialState = {
                field.name: columns[field.name][row_idx]
                for field in self._partial_fields
                if field.name is not None
            }
            accumulator = merge_partial_states(
                accumulator,
                row_partial,
                include_digest=self._include_digest,
            )
        return accumulator

    def combine(
        self,
        current_accumulator: PartialState | None,
        new: PartialState | None,
    ) -> PartialState | None:
        return merge_partial_states(
            current_accumulator,
            new,
            include_digest=self._include_digest,
        )

    def finalize(self, accumulator: PartialState | None) -> int | float | None:  # type: ignore[override]
        return self._stat.finalize_output(accumulator)


def required_partial_fields(resolved: ResolvedStats) -> tuple[Stat, ...]:
    required_names: set[str] = set()
    for name in resolved.requested:
        stat = _stat_for_name(name)
        if stat is None:
            continue
        required_names.update(
            field.name for field in _stat_partial_fields(stat) if field.name is not None
        )

    return tuple(
        field
        for field in PARTIAL_STAT_REGISTRY_BY_CLASS.values()
        if field.name is not None and field.name in required_names
    )


def required_partial_columns(resolved: ResolvedStats) -> tuple[str, ...]:
    return tuple(
        field.name for field in required_partial_fields(resolved) if field.name is not None
    )


def build_aggregations(resolved: ResolvedStats) -> list[AggregateFnV2]:
    aggs: list[AggregateFnV2] = []
    for name in resolved.requested:
        stat = _stat_for_name(name)
        if stat is not None:
            aggs.append(PartialStateAgg(stat))
    return aggs


# Backward-compatible helper used in tests.
def _finalize_quantile(state: PartialState | None, quantile: float) -> float:
    return QuantileStat(quantile=quantile).finalize_output(state)
