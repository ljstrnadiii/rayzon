from __future__ import annotations

import logging

import pyproj

from rayzon import GeoParquetDatasink, zonal_stats

AEF_STORE_URI = "s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic/"
AEF_ARRAY_NAME = "embeddings"
AEF_CRS = pyproj.CRS("EPSG:4326")
OUTPUT_PATH = "..."
INPUT_GEOPARQUET = "..."


def main() -> None:
    import ray

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ray.init(
        address="auto",
        ignore_reinit_error=True,
        log_to_driver=True,
        logging_level=logging.INFO,
    )

    zonal_stats(
        AEF_STORE_URI,
        INPUT_GEOPARQUET,
        array_name=AEF_ARRAY_NAME,
        stats=("mean", "count"),
        storage_options={"anon": True, "region": "us-west-2"},
        decode_coords=True,
        shuffle_num_partitions=64,
        feature_override_num_blocks=64,
        coord_columns={"time": "Y"},
        vectorize_dim="band",
        append_stats=True,
    ).write_datasink(
        GeoParquetDatasink(
            OUTPUT_PATH,
            crs=AEF_CRS.to_json_dict(),
        )
    )

    print(f"Saved result to {OUTPUT_PATH}")


if __name__ == "__main__":
    """
    ```
    ray job submit \
        --address http://127.0.0.1:8265 \
        --working-dir . \
        --runtime-env-json '{
            "env_vars": {
            "PYTHONPATH": "src"
            }
        }' \
        -- python examples/aef_example.py
    ```
    """
    main()
