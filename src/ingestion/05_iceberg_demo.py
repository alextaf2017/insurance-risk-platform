"""Demo funzioni Iceberg (tabella sandbox): snapshot, time travel, schema evolution,
delete, rollback e compattazione dei file."""
from common.spark_session import CATALOG, ensure_namespaces, get_spark

spark = get_spark("iceberg-demo")
ensure_namespaces(spark)
t = f"{CATALOG}.sandbox.demo_policies"

spark.sql(f"DROP TABLE IF EXISTS {t} PURGE")
(
    spark.table(f"{CATALOG}.gold.policy_features")
    .select("id_pol", "area", "driv_age", "claim_nb", "exposure")
    .limit(5000)
    .writeTo(t).using("iceberg").create()
)
snap1 = spark.sql(f"SELECT snapshot_id FROM {t}.snapshots ORDER BY committed_at ASC LIMIT 1").first()[0]
print(f"== Snapshot iniziale: {snap1}")

print("== DELETE righe area=A (nuovo snapshot)")
spark.sql(f"DELETE FROM {t} WHERE area = 'A'")

print("== Schema evolution: ADD COLUMN senza riscrivere i dati")
spark.sql(f"ALTER TABLE {t} ADD COLUMN risk_score DOUBLE")

print("== Storia degli snapshot")
spark.sql(f"SELECT snapshot_id, operation, committed_at FROM {t}.snapshots ORDER BY committed_at").show(truncate=False)

print("== Time travel: righe ora vs righe allo snapshot iniziale")
now = spark.sql(f"SELECT count(*) FROM {t}").first()[0]
before = spark.sql(f"SELECT count(*) FROM {t} VERSION AS OF {snap1}").first()[0]
print(f"ora={now}  snapshot_iniziale={before}")

print("== Rollback allo snapshot iniziale")
spark.sql(f"CALL {CATALOG}.system.rollback_to_snapshot('sandbox.demo_policies', {snap1})")
print("righe dopo rollback:", spark.sql(f"SELECT count(*) FROM {t}").first()[0])

print("== File prima/dopo compattazione")
print("prima:", spark.sql(f"SELECT count(*) FROM {t}.files").first()[0])
spark.sql(f"CALL {CATALOG}.system.rewrite_data_files(table => 'sandbox.demo_policies')")
print("dopo :", spark.sql(f"SELECT count(*) FROM {t}.files").first()[0])
