from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from rasterio.features import rasterize
from shapely.geometry.base import BaseGeometry

from rzzs.types import AffineLike


def rasterize_polygon_window(
    polygon: BaseGeometry,
    out_shape: tuple[int, int],
    transform: AffineLike,
    all_touched: bool = False,
) -> NDArray[np.bool_]:
    mask = rasterize(
        [(polygon, 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    )
    return cast(NDArray[np.bool_], np.asarray(mask, dtype=bool))
