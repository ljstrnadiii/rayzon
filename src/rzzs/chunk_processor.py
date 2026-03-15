from __future__ import annotations

from rzzs.config import ZonalStatsConfig
from rzzs.grid import GridSpec
from rzzs.types import ChunkId, FeatureId, GeometryStore, MosaicHandle, PartialRow


def process_chunk(
    chunk_id: ChunkId,
    feature_ids: list[FeatureId],
    mosaic_handle: MosaicHandle,
    geometry_store: GeometryStore,
    grid_spec: GridSpec,
    config: ZonalStatsConfig,
) -> list[PartialRow]:
    raise NotImplementedError("process_chunk will be implemented after grid/index foundations")
