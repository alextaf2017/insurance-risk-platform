"""Milestone 4, passo 1: reclami NHTSA nel lake (bronze, silver, gold) ed export per il container NLP.

bronze  lake.bronze.nhtsa_complaints   estratto con i soli campi necessari + metadati di ingestion
silver  lake.silver.nhtsa_complaints   un record per reclamo-componente, tipi corretti, narrativa ripulita
gold    lake.gold.nhtsa_narratives     un record per reclamo, etichetta `serious`, split temporale
export  data/nlp/narratives            Parquet per il fine-tuning (container nlp)

Variabili: RAW_DIR, NLP_DIR (default data/nlp), NHTSA_TRAIN_END (2023-12-31), NHTSA_VAL_END (2024-12-31).
"""
import os
import uuid
from pathlib import Path

from nhtsa_transforms import to_gold, to_silver
from pyspark.sql import functions as F

from common.dq import assert_rules
from common.spark_session import CATALOG, ensure_namespaces, get_spark

RAW = Path(os.getenv("RAW_DIR", "data/raw")) / "nhtsa" / "complaints.parquet"
OUT = Path(os.getenv("NLP_DIR", "data/nlp")).resolve()
TRAIN_END = os.getenv("NHTSA_TRAIN_END", "2023-12-31")
VAL_END = os.getenv("NHTSA_VAL_END", "2024-12-31")

spark = get_spark("nhtsa-lake")
ensure_namespaces(spark)

# ------------------------------------------------------------------ bronze
bronze = (
    spark.read.parquet(f"file://{RAW.resolve()}")
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source_file", F.lit(RAW.name))
    .withColumn("_batch_id", F.lit(uuid.uuid4().hex))
)
(bronze.writeTo(f"{CATALOG}.bronze.nhtsa_complaints").using("iceberg")
 .tableProperty("write.parquet.compression-codec", "zstd").createOrReplace())
n_bronze = spark.table(f"{CATALOG}.bronze.nhtsa_complaints").count()
print(f"[bronze] righe: {n_bronze:,}")

# ------------------------------------------------------------------ silver
silver = to_silver(spark.table(f"{CATALOG}.bronze.nhtsa_complaints"))
(silver.writeTo(f"{CATALOG}.silver.nhtsa_complaints").using("iceberg")
 .tableProperty("write.parquet.compression-codec", "zstd").createOrReplace())
silver_t = spark.table(f"{CATALOG}.silver.nhtsa_complaints")
assert_rules(
    silver_t, "silver.nhtsa_complaints",
    {
        "vehicle_only": "prod_type = 'V'",
        "ldate_not_null": "ldate IS NOT NULL",
        "text_min_length": "length(text) >= 30",
        "counts_non_negative": "injured >= 0 AND deaths >= 0",
    },
    unique_key="cmplid",
)
print(f"[silver] righe: {silver_t.count():,} (scartate rispetto al bronze: prodotti non veicolo, date o testi non validi)")

# ------------------------------------------------------------------ gold
gold = to_gold(spark.table(f"{CATALOG}.silver.nhtsa_complaints"), TRAIN_END, VAL_END)
(gold.writeTo(f"{CATALOG}.gold.nhtsa_narratives").using("iceberg")
 .partitionedBy(F.col("year"))
 .tableProperty("write.parquet.compression-codec", "zstd").createOrReplace())
gold_t = spark.table(f"{CATALOG}.gold.nhtsa_narratives")
assert_rules(
    gold_t, "gold.nhtsa_narratives",
    {
        "label_binary": "serious IN (0, 1)",
        "split_domain": "split IN ('train', 'validation', 'test')",
        "min_words": "n_words >= 5",
        "text_not_null": "text IS NOT NULL",
    },
    unique_key="odino",
)

print("\n== Gold: reclami e tasso di esito grave per split")
spark.sql(f"""
    SELECT split, count(*) AS complaints, round(avg(serious), 4) AS serious_rate,
           round(avg(n_words), 1) AS avg_words, min(ldate) AS from_date, max(ldate) AS to_date
    FROM {CATALOG}.gold.nhtsa_narratives GROUP BY split ORDER BY split
""").show(truncate=False)

print("== Controllo di coerenza: le sigle dell'analista in coda sono associate all'etichetta?")
spark.sql(f"""
    SELECT had_initials, count(*) AS complaints, round(avg(serious), 4) AS serious_rate
    FROM {CATALOG}.gold.nhtsa_narratives GROUP BY had_initials ORDER BY had_initials
""").show()

print("== Controllo di coerenza: l'etichetta di origine in testa (es. 'TL*') e' associata all'etichetta?")
spark.sql(f"""
    SELECT had_prefix, count(*) AS complaints, round(avg(serious), 4) AS serious_rate
    FROM {CATALOG}.gold.nhtsa_narratives GROUP BY had_prefix ORDER BY had_prefix
""").show()

print("== Esito grave: da quali campi e' composto (non esclusivi)")
spark.sql(f"""
    SELECT sum(CASE WHEN injured > 0 THEN 1 ELSE 0 END) AS injured,
           sum(deaths) AS deaths_total, sum(medical_attn) AS medical_attn,
           sum(serious) AS serious_total, count(*) AS complaints
    FROM {CATALOG}.gold.nhtsa_narratives
""").show()

# ------------------------------------------------------------------ export per il container NLP
cols = ["odino", "ldate", "year", "make", "model", "model_year", "components", "text", "n_words",
        "serious", "injured", "deaths", "medical_attn", "crash", "fire", "towed", "had_initials", "had_prefix", "split"]
OUT.mkdir(parents=True, exist_ok=True)
gold_t.select(*cols).coalesce(1).write.mode("overwrite").parquet(f"file://{OUT}/narratives")
print(f"[export] {OUT}/narratives")
spark.stop()
