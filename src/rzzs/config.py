from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ZonalStatsConfig:
    x_dim: str
    y_dim: str
    all_touched: bool = False
    nodata: float | int | None = None
    dst_crs: str | None = None
    quantiles: tuple[float, ...] = ()
    quantile_compression: int = 200
    exact_quantile_pixel_threshold: int = 100_000

    def __post_init__(self) -> None:
        if not self.x_dim or not self.y_dim:
            raise ValueError("x_dim and y_dim are required")
        if self.quantile_compression <= 0:
            raise ValueError("quantile_compression must be > 0")
        if self.exact_quantile_pixel_threshold < 0:
            raise ValueError("exact_quantile_pixel_threshold must be >= 0")
        for q in self.quantiles:
            if not (0.0 <= q <= 1.0):
                raise ValueError("quantiles must be in [0, 1]")


@dataclass(frozen=True)
class BenchmarkConfig:
    num_polygons: int = 1000
    num_chunks_target: int = 100
    num_time_steps: int = 10
    repeat: int = 3

    def __post_init__(self) -> None:
        if self.num_polygons <= 0:
            raise ValueError("num_polygons must be > 0")
        if self.num_chunks_target <= 0:
            raise ValueError("num_chunks_target must be > 0")
        if self.num_time_steps <= 0:
            raise ValueError("num_time_steps must be > 0")
        if self.repeat <= 0:
            raise ValueError("repeat must be > 0")
