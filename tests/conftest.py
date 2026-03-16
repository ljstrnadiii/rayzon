from collections.abc import Generator

import pytest
import ray
import ray.data


@pytest.fixture(scope="session", autouse=True)
def ray_cluster() -> Generator[None, None, None]:
    # Start a single Ray cluster for the entire test session and configure
    # DataContext defaults so two back-to-back groupby shuffles don't deadlock
    # on a small machine.
    ray.init(num_cpus=2, ignore_reinit_error=True)
    ctx = ray.data.DataContext.get_current()
    ctx.min_parallelism = 1
    ctx.default_hash_shuffle_parallelism = 1
    yield
    ray.shutdown()
