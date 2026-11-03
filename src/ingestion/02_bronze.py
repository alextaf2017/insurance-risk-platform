"""BRONZE: dati grezzi cosi come arrivano + metadati di ingestion. Nessuna pulizia."""
import os
import uuid
from pathlib import Path

from pyspark.sql import functions as F

from common.spark_session import CATALOG, ensure_namespaces, get_spark

RAW = Path(os.getenv("RAW_DIR", "data/raw"))
TABLES = ["fremtpl2_freq", "fremtpl2_sev"]

spark = get_spark("bronze")
ensure_namespaces(spark)
batch_id = uuid.uuid4().hex

for name in TABLES:
    src = RAW / f"{name}.parquet"
    df = (
        spark.read.parquet(f"file://{src.resolve()}")
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_file", F.lit(src.name))
        .withColumn("_batch_id", F.lit(batch_id))
    )
    target = f"{CATALOG}.bronze.{name}"
    (
        df.writeTo(target)
        .using("iceberg")
        .tableProperty("format-version", "2")
        .tableProperty("write.parquet.compression-codec", "zstd")
        .createOrReplace()  # ogni run crea un nuovo snapshot: la storia resta interrogabile
    )
    print(f"[bronze] {target}: {spark.table(target).count():,} righe")
