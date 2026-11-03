"""SILVER: tipi corretti, nomi snake_case, deduplica, regole di validita e quarantena.

Anomalie note su freMTPL2 (Noll, Salzmann, Wuthrich): Exposure > 1 e ClaimNb molto alto.
Si tiene il valore originale (_raw), si applica il cap e si segnala con un flag.
"""
from pyspark.sql import functions as F

from common.dq import assert_rules
from common.spark_session import CATALOG, ensure_namespaces, get_spark

spark = get_spark("silver")
ensure_namespaces(spark)

# ---------------- Frequenza ----------------
b = spark.table(f"{CATALOG}.bronze.fremtpl2_freq")

typed = b.select(
    F.col("IDpol").cast("double").cast("long").alias("id_pol"),
    F.col("ClaimNb").cast("double").cast("int").alias("claim_nb_raw"),
    F.col("Exposure").cast("double").alias("exposure_raw"),
    F.upper(F.trim("Area")).alias("area"),
    F.col("VehPower").cast("double").cast("int").alias("veh_power"),
    F.col("VehAge").cast("double").cast("int").alias("veh_age"),
    F.col("DrivAge").cast("double").cast("int").alias("driv_age"),
    F.col("BonusMalus").cast("double").cast("int").alias("bonus_malus"),
    F.upper(F.trim("VehBrand")).alias("veh_brand"),
    F.initcap(F.trim("VehGas")).alias("veh_gas"),
    F.col("Density").cast("double").cast("int").alias("density"),
    F.upper(F.trim("Region")).alias("region"),
    "_batch_id",
)

# Deduplica su chiave di business
typed = typed.dropDuplicates(["id_pol"])

reason = (
    F.when(F.col("id_pol").isNull(), "id_pol_null")
    .when(F.col("exposure_raw").isNull() | (F.col("exposure_raw") <= 0), "exposure_non_positive")
    .when(F.col("claim_nb_raw").isNull() | (F.col("claim_nb_raw") < 0), "claim_nb_invalid")
    .when(F.col("driv_age") < 18, "driver_underage")
)
flagged = typed.withColumn("_reject_reason", reason)

rejected = flagged.filter(F.col("_reject_reason").isNotNull())
valid = (
    flagged.filter(F.col("_reject_reason").isNull())
    .drop("_reject_reason")
    .withColumn("exposure", F.least(F.col("exposure_raw"), F.lit(1.0)))
    .withColumn("claim_nb", F.least(F.col("claim_nb_raw"), F.lit(4)))
    .withColumn("is_exposure_capped", F.col("exposure_raw") > 1.0)
    .withColumn("is_claim_nb_capped", F.col("claim_nb_raw") > 4)
)

valid.writeTo(f"{CATALOG}.silver.fremtpl2_freq").using("iceberg").createOrReplace()
rejected.writeTo(f"{CATALOG}.silver.fremtpl2_freq_rejected").using("iceberg").createOrReplace()

assert_rules(
    spark.table(f"{CATALOG}.silver.fremtpl2_freq"),
    "silver.fremtpl2_freq",
    {
        "id_pol_not_null": "id_pol IS NOT NULL",
        "exposure_in_range": "exposure > 0 AND exposure <= 1",
        "claim_nb_range": "claim_nb BETWEEN 0 AND 4",
        "area_domain": "area IN (\"A\",\"B\",\"C\",\"D\",\"E\",\"F\")",
        "driv_age_adult": "driv_age >= 18",
    },
    unique_key="id_pol",
)
n_ok = spark.table(f"{CATALOG}.silver.fremtpl2_freq").count()
n_ko = spark.table(f"{CATALOG}.silver.fremtpl2_freq_rejected").count()
print(f"[silver] freq validi={n_ok:,} scartati={n_ko:,}")

# ---------------- Severita (una riga per sinistro) ----------------
s = (
    spark.table(f"{CATALOG}.bronze.fremtpl2_sev")
    .select(
        F.col("IDpol").cast("double").cast("long").alias("id_pol"),
        F.col("ClaimAmount").cast("double").alias("claim_amount"),
        "_batch_id",
    )
    .filter(F.col("id_pol").isNotNull() & (F.col("claim_amount") > 0))
)
s.writeTo(f"{CATALOG}.silver.fremtpl2_sev").using("iceberg").createOrReplace()

assert_rules(
    spark.table(f"{CATALOG}.silver.fremtpl2_sev"),
    "silver.fremtpl2_sev",
    {"id_pol_not_null": "id_pol IS NOT NULL", "amount_positive": "claim_amount > 0"},
)
