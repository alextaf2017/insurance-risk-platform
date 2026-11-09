"""Milestone 3, passo 1: esporta dal lake i dati di modeling in Parquet (train e test, con fold).

L'esportazione usa Spark; i modelli girano poi in Python puro (senza JVM) per non saturare la memoria.
Output in data/modeling/ (ignorata da git).
"""
import os
from pathlib import Path

from common.spark_session import get_spark

OUT = Path(os.getenv("MODELING_DIR", "data/modeling")).resolve()
OUT.mkdir(parents=True, exist_ok=True)

spark = get_spark("export-modeling")

policies = spark.sql("""
    SELECT id_pol, split, fold, claim_nb, exposure, total_claim_amount, sev_claim_count,
           area, driv_age_c, veh_age_c, veh_power, bonus_malus_c, log_density,
           veh_gas, veh_brand, region
    FROM lake.gold.policy_features
""")
claims = spark.sql("""
    SELECT s.id_pol, s.claim_amount, g.split, g.fold,
           g.area, g.driv_age_c, g.veh_age_c, g.veh_power, g.bonus_malus_c, g.log_density,
           g.veh_gas, g.veh_brand, g.region
    FROM lake.silver.fremtpl2_sev s
    JOIN lake.gold.policy_features g ON s.id_pol = g.id_pol
""")

policies.coalesce(1).write.mode("overwrite").parquet(f"file://{OUT}/policies")
claims.coalesce(1).write.mode("overwrite").parquet(f"file://{OUT}/claims")
print(f"[export] policies={policies.count():,} claims={claims.count():,} -> {OUT}")
spark.stop()
