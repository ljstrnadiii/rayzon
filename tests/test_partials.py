import math

import numpy as np

from rzzs.partials import compute_partials, finalize_moments


def test_compute_partials_with_mask_and_nodata() -> None:
    values = np.array([[1.0, np.nan], [3.0, 99.0]])
    mask = np.array([[True, True], [True, True]])
    out = compute_partials(values, mask, nodata=99.0, need_sum_sq=True)
    assert out["count"] == 4
    assert out["n_valid"] == 2
    assert out["sum"] == 4.0
    assert out["sum_sq"] == 10.0


def test_finalize_moments_zero_count() -> None:
    out = finalize_moments(0, 0.0, 0.0)
    assert math.isnan(out["mean"])
