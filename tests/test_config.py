import pytest

from rzzs.config import BenchmarkConfig, ZonalStatsConfig


def test_zonal_stats_config_validates_quantiles() -> None:
    with pytest.raises(ValueError):
        ZonalStatsConfig(x_dim="x", y_dim="y", quantiles=(1.2,))


def test_benchmark_config_positive_values() -> None:
    with pytest.raises(ValueError):
        BenchmarkConfig(num_polygons=0)
