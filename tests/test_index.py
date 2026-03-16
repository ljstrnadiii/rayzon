from rzzs.grid import GridSpec
from rzzs.index import build_feature_chunk_index_from_chunk_to_features, plan_chunk_jobs


def _grid() -> GridSpec:
    return GridSpec(
        dims=("y", "x"),
        shape=(100, 100),
        chunk_sizes=(10, 10),
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 100.0),
        crs="EPSG:4326",
        x_dim="x",
        y_dim="y",
    )


def test_build_feature_chunk_index_from_chunk_to_features() -> None:
    chunk_to_features: dict[tuple[int, ...], list[int | str]] = {
        (0, 0): ["a", "b"],
        (0, 1): ["a"],
    }

    index = build_feature_chunk_index_from_chunk_to_features(chunk_to_features)
    assert "a" in index.feature_to_chunks
    assert len(index.feature_to_chunks["a"]) == 2
    assert len(index.feature_to_chunks["b"]) == 1


def test_plan_chunk_jobs_sorted() -> None:
    chunk_to_features: dict[tuple[int, ...], list[int | str]] = {
        (0, 1): ["a"],
        (0, 0): ["a"],
    }
    index = build_feature_chunk_index_from_chunk_to_features(chunk_to_features)
    jobs = plan_chunk_jobs(index)
    chunk_ids = [job["chunk_id"] for job in jobs]
    assert chunk_ids == sorted(chunk_ids)
