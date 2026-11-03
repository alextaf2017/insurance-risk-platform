"""Esplorazione rapida del lake: tabelle, snapshot, file fisici, anomalie."""
from common.spark_session import get_spark

spark = get_spark("explore")

print("== Tabelle nel lake")
for ns in ["bronze", "silver", "gold"]:
    spark.sql(f"SHOW TABLES IN lake.{ns}").show(truncate=False)

print("== Snapshot della tabella bronze (uno per ogni run)")
spark.sql("SELECT snapshot_id, operation, committed_at FROM lake.bronze.fremtpl2_freq.snapshots").show(truncate=False)

print("== File fisici della tabella gold (partizionata per area)")
spark.sql("SELECT partition, record_count, file_size_in_bytes FROM lake.gold.policy_features.files ORDER BY partition").show(truncate=False)

print("== Righe corrette (cap) in silver")
spark.sql("""
    SELECT sum(CASE WHEN is_exposure_capped THEN 1 ELSE 0 END) AS exposure_capped,
           sum(CASE WHEN is_claim_nb_capped THEN 1 ELSE 0 END) AS claim_nb_capped
    FROM lake.silver.fremtpl2_freq
""").show()

print("== Tipologia delle incoerenze frequenza/severità")
spark.sql("""
    SELECT claim_nb > 0 AS has_claim_nb, sev_claim_count > 0 AS has_sev_record, count(*) AS n
    FROM lake.gold.policy_features
    WHERE is_claim_inconsistent
    GROUP BY 1, 2
""").show()
