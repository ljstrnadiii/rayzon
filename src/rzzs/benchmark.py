from __future__ import annotations

from rzzs.config import BenchmarkConfig


def benchmark_pipeline(config: BenchmarkConfig) -> dict[str, float | int]:
    raise NotImplementedError("benchmark pipeline not implemented yet")
