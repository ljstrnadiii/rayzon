from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_STAT_EXPRS: tuple[str, ...] = ("count", "n_valid", "mean", "std")


@dataclass(frozen=True)
class ResolvedStats:
    requested: tuple[str, ...]
    quantiles: tuple[float, ...]
    need_count: bool
    need_n_valid: bool
    need_sum: bool
    need_sum_sq: bool
    need_min: bool
    need_max: bool


def normalize_stat_expr(expr: str) -> str:
    builtin_stats = {
        "count",
        "n_valid",
        "sum",
        "min",
        "max",
        "mean",
        "variance",
        "std",
    }
    quantile_re = re.compile(r"^p_?(100|[0-9]{1,2})$")

    normalized = expr.strip().lower().replace(" ", "")
    if normalized in builtin_stats:
        return normalized

    match = quantile_re.fullmatch(normalized)
    if match is not None:
        q = int(match.group(1))
        return f"p{q}"

    raise ValueError(
        f"Unsupported stat expression '{expr}'. Use builtins ({sorted(builtin_stats)}) "
        "or quantiles like 'p50'/'p_73'."
    )


def resolve_stat_exprs(stats: tuple[str, ...] | list[str]) -> ResolvedStats:
    normalized = tuple(dict.fromkeys(normalize_stat_expr(expr) for expr in stats))
    quantiles = tuple(int(item[1:]) / 100.0 for item in normalized if item.startswith("p"))

    need_sum_sq = "variance" in normalized or "std" in normalized
    need_sum = need_sum_sq or "sum" in normalized or "mean" in normalized
    need_count = "count" in normalized
    need_n_valid = (
        "n_valid" in normalized
        or "mean" in normalized
        or "variance" in normalized
        or "std" in normalized
        or "sum" in normalized
        or any(item.startswith("p") for item in normalized)
    )
    need_min = "min" in normalized
    need_max = "max" in normalized

    return ResolvedStats(
        requested=normalized,
        quantiles=quantiles,
        need_count=need_count,
        need_n_valid=need_n_valid,
        need_sum=need_sum,
        need_sum_sq=need_sum_sq,
        need_min=need_min,
        need_max=need_max,
    )
