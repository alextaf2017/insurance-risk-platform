"""Milestone 2B, passo 1: esporta dal lake i dati di TRAIN in Parquet per l'analisi in Python.

Separare l'esportazione (Spark) dall'analisi (statsmodels) evita di tenere la JVM in memoria
mentre si stimano i GLM. Output in data/analysis/ (ignorata da git).
"""
import os
from pathlib import Path

from common.spark_session import get_spark

OUT = Path(os.getenv("ANALYSIS_DIR", "data/analysis")).resolve()
OUT.mkdir(parents=True, exist_ok=True)

spark = get_spark("export-train")

policies = spark.sql("""
    SELECT id_pol, claim_nb, exposure, veh_power, veh_age_c, driv_age_c, bonus_malus_c, log_density,
           area, driv_age_band, veh_gas, veh_brand, region
    FROM lake.gold.policy_features
    WHERE split = 'train'
""")
claims = spark.sql("""
    SELECT s.claim_amount, g.area, g.driv_age_band, g.bonus_malus_c, g.veh_age_c,
           g.veh_power, g.veh_gas, g.log_density
    FROM lake.silver.fremtpl2_sev s
    JOIN lake.gold.policy_features g ON s.id_pol = g.id_pol
    WHERE g.split = 'train'
""")

policies.coalesce(1).write.mode("overwrite").parquet(f"file://{OUT}/policies_train")
claims.coalesce(1).write.mode("overwrite").parquet(f"file://{OUT}/claims_train")
print(f"[export] policies={policies.count():,} claims={claims.count():,} -> {OUT}")
spark.stop()
