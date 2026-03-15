import pytest

from rzzs.stats import normalize_stat_expr, resolve_stat_exprs


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
    assert resolved.need_sum is True
    assert resolved.need_sum_sq is True
    assert resolved.need_n_valid is True


def test_resolve_stat_exprs_count_only() -> None:
    resolved = resolve_stat_exprs(["count"])
    assert resolved.need_count is True
    assert resolved.need_n_valid is False
    assert resolved.need_sum is False
