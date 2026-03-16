from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from shapely.geometry import box

from rzzs.rasterize_backend import RasterizeBackend, rasterize_polygon_window


def test_rasterize_polygon_window_accepts_numpy_transform() -> None:
    polygon = box(0.0, 0.0, 1.0, 1.0)
    transform = np.array([1.0, 0.0, 0.0, 0.0, -1.0, 1.0], dtype=np.float64)

    mask = rasterize_polygon_window(polygon, out_shape=(2, 2), transform=transform)

    assert mask.dtype == np.bool_
    assert mask.shape == (2, 2)
    assert int(mask.sum()) == 1


def test_rasterize_polygon_window_rejects_bad_transform_size() -> None:
    polygon = box(0.0, 0.0, 1.0, 1.0)
    bad_transform = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="exactly 6 values"):
        rasterize_polygon_window(polygon, out_shape=(2, 2), transform=bad_transform)


def test_rasterize_polygon_window_rejects_unknown_backend() -> None:
    polygon = box(0.0, 0.0, 1.0, 1.0)
    transform = np.array([1.0, 0.0, 0.0, 0.0, -1.0, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="Unsupported rasterize backend"):
        rasterize_polygon_window(
            polygon,
            out_shape=(2, 2),
            transform=transform,
            rasterize_backend=cast(RasterizeBackend, "unknown"),
        )
