import random

import numpy as np
import pyarrow as pa
import pytest

from rayzon.stats import (
    _STAT_REGISTRY,
    _finalize_quantile,
    accumulate_partial_state,
    build_aggregations,
    merge_partial_states,
    normalize_stat_expr,
    partial_column_arrow_type,
    partial_fields_for_columns,
    required_partial_columns,
    resolve_stat_exprs,
)

COL_DIGEST = "_partial_digest"
COL_MIN = "_partial_min"
COL_MAX = "_partial_max"
COL_N_VALID = "_partial_n_valid"
COL_SUM = "_partial_sum"
COL_SUM_SQ = "_partial_sum_sq"


def test_normalize_stat_expr_builtin() -> None:
    assert normalize_stat_expr(" Mean ") == "mean"


def test_normalize_stat_expr_quantile() -> None:
    assert normalize_stat_expr("p_73") == "p73"


def test_normalize_stat_expr_invalid() -> None:
    with pytest.raises(ValueError):
        normalize_stat_expr("medianish")


def test_resolve_stat_exprs_dependencies() -> None:
    resolved = resolve_stat_exprs(["mean", "std", "p_73", "n_valid"])
    assert resolved.requested == ("mean", "std", "p73", "n_valid")
    assert resolved.quantiles == (0.73,)


def test_resolve_stat_exprs_count_only() -> None:
    resolved = resolve_stat_exprs(["count"])
    assert resolved.quantiles == ()


def test_required_partial_columns_mean_only() -> None:
    resolved = resolve_stat_exprs(["mean"])
    assert required_partial_columns(resolved) == (COL_N_VALID, COL_SUM)


def test_required_partial_columns_std_and_quantiles() -> None:
    resolved = resolve_stat_exprs(["std", "p50", "p95"])
    assert required_partial_columns(resolved) == (
        COL_N_VALID,
        COL_SUM,
        COL_SUM_SQ,
        COL_DIGEST,
    )


def test_partial_fields_for_columns_dedup_and_order() -> None:
    fields = partial_fields_for_columns(
        (
            COL_DIGEST,
            COL_N_VALID,
            COL_DIGEST,
            COL_SUM,
        )
    )
    assert tuple(field.name for field in fields) == (
        COL_DIGEST,
        COL_N_VALID,
        COL_SUM,
    )


def test_partial_fields_for_columns_invalid() -> None:
    with pytest.raises(ValueError):
        partial_fields_for_columns(("nope",))


def test_partial_column_arrow_type_known_and_invalid() -> None:
    assert partial_column_arrow_type(COL_MIN) == pa.float64()
    assert partial_column_arrow_type(COL_MAX) == pa.float64()
    with pytest.raises(ValueError):
        partial_column_arrow_type("unknown_col")


def test_accumulate_partial_state_returns_none_for_empty_inputs() -> None:
    assert (
        accumulate_partial_state(
            np.array([], dtype=np.float64),
            required_fields=partial_fields_for_columns((COL_SUM,)),
        )
        is None
    )
    assert (
        accumulate_partial_state(
            np.array([1.0, 2.0], dtype=np.float64),
            required_fields=(),
        )
        is None
    )


def test_merge_partial_states_respects_include_digest_flag() -> None:
    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    digest_only = accumulate_partial_state(
        values,
        required_fields=partial_fields_for_columns((COL_DIGEST,)),
    )
    assert digest_only is not None

    dropped = merge_partial_states(None, digest_only, include_digest=False)
    assert dropped == {}

    kept = merge_partial_states(None, digest_only, include_digest=True)
    assert kept is not None
    assert COL_DIGEST in kept


def test_build_aggregations_preserves_requested_order() -> None:
    resolved = resolve_stat_exprs(["mean", "p95", "count"])
    aggs = build_aggregations(resolved)
    assert [agg.name for agg in aggs] == ["mean", "p95", "count"]


@pytest.mark.parametrize("stat_name", sorted(_STAT_REGISTRY))
def test_registered_stat_merge_matches_single_pass(stat_name: str) -> None:
    values = np.array([1.5, np.nan, -2.0, 4.25, 0.0, np.nan, 8.0], dtype=np.float64)
    blocks = (values[:3], values[3:5], values[5:])
    stat = _STAT_REGISTRY[stat_name]
    resolved = resolve_stat_exprs([stat_name])
    partial_fields = partial_fields_for_columns(required_partial_columns(resolved))
    include_digest = any(field.name == COL_DIGEST for field in partial_fields)

    merged_state = None
    for block in blocks:
        partial = accumulate_partial_state(block, required_fields=partial_fields)
        merged_state = merge_partial_states(
            merged_state,
            partial,
            include_digest=include_digest,
        )

    single_state = accumulate_partial_state(values, required_fields=partial_fields)
    merged_output = stat.finalize_output(merged_state)
    single_output = stat.finalize_output(single_state)
    np.testing.assert_allclose(
        float(merged_output),  # type: ignore[arg-type]
        float(single_output),  # type: ignore[arg-type]
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )


def _digest_quantiles_from_blocks(
    values: np.ndarray,
    *,
    quantiles: tuple[float, ...],
    block_seed: int,
) -> dict[float, float]:
    block_rng = random.Random(block_seed)
    block_sizes = [block_rng.randint(80, 500) for _ in range(100)]

    state = None
    index = 0
    for block_size in block_sizes:
        if index >= len(values):
            break
        block = values[index : index + block_size]
        index += block_size
        partial = accumulate_partial_state(
            block,
            required_fields=partial_fields_for_columns((COL_DIGEST,)),
        )
        state = merge_partial_states(state, partial, include_digest=True)

    if index < len(values):
        partial = accumulate_partial_state(
            values[index:],
            required_fields=partial_fields_for_columns((COL_DIGEST,)),
        )
        state = merge_partial_states(state, partial, include_digest=True)

    return {q: float(_finalize_quantile(state, q)) for q in quantiles}


def _rank_error(values: np.ndarray, estimate: float, quantile: float) -> float:
    empirical_rank = float(np.mean(values <= estimate))
    return abs(empirical_rank - quantile)


def test_tdigest_quantiles_gaussian_blocks_are_close() -> None:
    rng = np.random.default_rng(9)
    values = rng.normal(0.0, 1.0, size=12_000).astype(np.float64)
    quantiles = (0.5, 0.95, 0.99)

    random.seed(123)
    estimated = _digest_quantiles_from_blocks(values, quantiles=quantiles, block_seed=123)

    for q in quantiles:
        exact = float(np.quantile(values, q))
        rank_error = _rank_error(values, estimated[q], q)
        np.testing.assert_allclose(rank_error, 0.0, atol=0.003, rtol=0.0)
        np.testing.assert_allclose(estimated[q], exact, rtol=0.02, atol=1e-3)


def test_tdigest_quantiles_skewed_blocks_are_close() -> None:
    rng = np.random.default_rng(9)
    base = rng.lognormal(mean=0.0, sigma=1.2, size=10_800)
    tail = rng.pareto(a=2.0, size=1_200) * 20.0
    values = np.concatenate([base, tail]).astype(np.float64)
    quantiles = (0.5, 0.95, 0.99)

    random.seed(321)
    estimated = _digest_quantiles_from_blocks(values, quantiles=quantiles, block_seed=321)

    for q in quantiles:
        exact = float(np.quantile(values, q))
        rank_error = _rank_error(values, estimated[q], q)
        np.testing.assert_allclose(rank_error, 0.0, atol=0.004, rtol=0.0)
        np.testing.assert_allclose(estimated[q], exact, rtol=0.02, atol=0.0)
