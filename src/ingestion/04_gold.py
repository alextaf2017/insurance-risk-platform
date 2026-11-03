"""GOLD: data product policy_features, pronto per il modeling (un record per polizza)."""
from pyspark.sql import functions as F

from common.dq import assert_rules
from common.spark_session import CATALOG, ensure_namespaces, get_spark

spark = get_spark("gold")
ensure_namespaces(spark)

freq = spark.table(f"{CATALOG}.silver.fremtpl2_freq")
sev = (
    spark.table(f"{CATALOG}.silver.fremtpl2_sev")
    .groupBy("id_pol")
    .agg(
        F.count("*").alias("sev_claim_count"),
        F.sum("claim_amount").alias("total_claim_amount"),
    )
)

gold = (
    freq.join(sev, "id_pol", "left")
    .withColumn("sev_claim_count", F.coalesce("sev_claim_count", F.lit(0)))
    .withColumn("total_claim_amount", F.coalesce("total_claim_amount", F.lit(0.0)))
    # Target
    .withColumn("claim_freq", F.col("claim_nb") / F.col("exposure"))
    .withColumn(
        "avg_severity",
        F.when(F.col("sev_claim_count") > 0, F.col("total_claim_amount") / F.col("sev_claim_count")),
    )
    # Feature engineering (cap come da letteratura attuariale)
    .withColumn("veh_age_c", F.least("veh_age", F.lit(20)))
    .withColumn("driv_age_c", F.least("driv_age", F.lit(90)))
    .withColumn("bonus_malus_c", F.least("bonus_malus", F.lit(150)))
    .withColumn("log_density", F.log("density"))
    .withColumn(
        "driv_age_band",
        F.when(F.col("driv_age") < 25, "18-24")
        .when(F.col("driv_age") < 35, "25-34")
        .when(F.col("driv_age") < 50, "35-49")
        .when(F.col("driv_age") < 65, "50-64")
        .otherwise("65+"),
    )
    # Incoerenza nota: ClaimNb>0 senza record di severita (o viceversa)
    .withColumn(
        "is_claim_inconsistent",
        (F.col("claim_nb") > 0) != (F.col("sev_claim_count") > 0),
    )
    # Split deterministico (hash dell id): stabile tra run, evita leakage tra train e test
    .withColumn("fold", F.pmod(F.xxhash64("id_pol"), F.lit(10)).cast("int"))
    .withColumn("split", F.when(F.col("fold") < 8, "train").otherwise("test"))
    .drop("claim_nb_raw", "exposure_raw", "_batch_id")
)

target = f"{CATALOG}.gold.policy_features"
(
    gold.writeTo(target)
    .using("iceberg")
    .partitionedBy(F.col("area"))  # partizionamento Iceberg
    .tableProperty("write.parquet.compression-codec", "zstd")
    .createOrReplace()
)

assert_rules(
    spark.table(target),
    "gold.policy_features",
    {
        "exposure_in_range": "exposure > 0 AND exposure <= 1",
        "freq_non_negative": "claim_freq >= 0",
        "amount_non_negative": "total_claim_amount >= 0",
        "split_domain": "split IN (\"train\",\"test\")",
    },
    unique_key="id_pol",
)

spark.sql(f"""
    SELECT split, count(*) AS policies,
           round(sum(claim_nb)/sum(exposure), 4) AS freq,
           sum(CASE WHEN is_claim_inconsistent THEN 1 ELSE 0 END) AS inconsistent
    FROM {target} GROUP BY split ORDER BY split
""").show()
