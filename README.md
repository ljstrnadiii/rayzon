# ray-zarr-zonal-stats

Proof-of-concept chunk-first zonal statistics for xarray mosaics and GeoPandas polygons using Ray.

## Development

- Install dependencies: `uv sync --extra dev --extra geo`
- Run lint: `uv run ruff check .`
- Run format check: `uv run ruff format --check .`
- Run type check: `uv run mypy src`
- Run tests: `uv run pytest`
