# `rayzon` (Ray Zonal Statistics on Zarr)
`rayzon` is a ray project to execute zonal statistics on zarr arrays.

It is designed for workflows where:
- Raster data is stored as chunked arrays with optional non-spatial dimensions e.g.
`(time, band, y, x)`.
- Vector features may overlap.
- Relevant raster chunks are read once, and statistics are computed in a distributed manner with Ray.


The project uses a chunk-first execution model:
1. Read a raster chunk once.
2. Find candidate features that intersect that chunk.
3. Rasterize only the local polygon window (not the full chunk unless required).
4. Emit pyarrow blocks of flattened pixel data per intersecting block and leverages Ray's
`AggregateFnV2` to group and reduce those blocks into final statistics.

This architecture keeps memory usage bounded, avoids repeated raster reads, and makes distributed
execution with Ray straightforward.

## Features

- **Zarr as the storage contract:** Zarr gives us chunked, cloud-friendly storage today, while
preserving flexibility to plug in VirtualiZarr-backed datasets later without changing the core
chunk-first pipeline.
- **Ray `AggregateFnV2` for extensible reductions:** Chunk work emits PyArrow blocks for features
intersecting each chunk, then Ray can `groupby` and reduce those blocks with `AggregateFnV2`. This
keeps reduction logic composable and makes new statistics easy to add as new aggregate definitions.
- **GeoArrow-native geometry transport:** GeoArrow lets us read and move geometry types directly in
PyArrow form, reducing conversion overhead and keeping feature transport efficient across Ray tasks.

## Current Scope

- Chunk/grid utilities for deterministic chunk addressing.
- Feature-to-chunk indexing.
- Windowed rasterization backend.
- Partial stats and reduction pipeline.
- End-to-end pipeline entry points and benchmark hooks.

## Development

- Install dependencies: `uv sync --extra dev --extra geo`
- Run lint: `uv run ruff check .`
- Run format check: `uv run ruff format --check .`
- Run type check: `uv run mypy src`
- Run tests: `uv run pytest`
