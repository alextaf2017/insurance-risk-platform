"""Milestone 2A - EDA con Spark SQL: CTE, window function e piani di esecuzione."""
import time

from common.spark_session import get_spark

spark = get_spark("eda-sql")
G = "lake.gold.policy_features"


def run(title, sql, n=20):
    print(f"\n== {title}")
    spark.sql(sql).show(n, truncate=False)


run("1. Portafoglio per area", f"""
    SELECT area,
           count(*) AS policies,
           round(sum(exposure), 0) AS exposure,
           sum(claim_nb) AS claims,
           round(sum(claim_nb) / sum(exposure), 4) AS frequency,
           round(sum(total_claim_amount) / nullif(sum(sev_claim_count), 0), 2) AS avg_severity
    FROM {G}
    GROUP BY area
    ORDER BY area
""")

run("2. Frequenza per fascia di età: relatività rispetto al portafoglio (CTE + window)", f"""
    WITH by_band AS (
        SELECT driv_age_band, sum(exposure) AS exposure, sum(claim_nb) AS claims
        FROM {G}
        GROUP BY driv_age_band
    )
    SELECT driv_age_band,
           round(exposure, 0) AS exposure,
           round(100 * exposure / sum(exposure) OVER (), 1) AS exposure_pct,
           round(claims / exposure, 4) AS frequency,
           round((claims / exposure) / (sum(claims) OVER () / sum(exposure) OVER ()), 3) AS relativity
    FROM by_band
    ORDER BY driv_age_band
""")

run("3. Regioni per frequenza, con esposizione cumulata (RANK + window cumulativa)", f"""
    WITH reg AS (
        SELECT region, sum(exposure) AS exposure, sum(claim_nb) AS claims
        FROM {G}
        GROUP BY region
        HAVING sum(exposure) >= 1000
    )
    SELECT region,
           round(exposure, 0) AS exposure,
           round(claims / exposure, 4) AS frequency,
           RANK() OVER (ORDER BY claims / exposure DESC) AS freq_rank,
           round(100 * sum(exposure) OVER (ORDER BY exposure DESC) / sum(exposure) OVER (), 1) AS cum_exposure_pct
    FROM reg
    ORDER BY freq_rank
""")

run("4. Frequenza per fascia di Bonus-Malus", f"""
    SELECT CASE WHEN bonus_malus_c <= 50 THEN '1: 50'
                WHEN bonus_malus_c <= 75 THEN '2: 51-75'
                WHEN bonus_malus_c <= 100 THEN '3: 76-100'
                WHEN bonus_malus_c <= 125 THEN '4: 101-125'
                ELSE '5: 126+' END AS bonus_malus_band,
           count(*) AS policies,
           round(sum(claim_nb) / sum(exposure), 4) AS frequency
    FROM {G}
    GROUP BY 1
    ORDER BY 1
""")

run("5. Coda pesante della severità: quota dell'importo totale nei sinistri più grandi", """
    WITH ranked AS (
        SELECT claim_amount,
               row_number() OVER (ORDER BY claim_amount DESC) AS rn,
               sum(claim_amount) OVER () AS total,
               count(*) OVER () AS n
        FROM lake.silver.fremtpl2_sev
    )
    SELECT round(100 * sum(CASE WHEN rn <= 0.01 * n THEN claim_amount END) / max(total), 1) AS top_1pct_share,
           round(100 * sum(CASE WHEN rn <= 0.10 * n THEN claim_amount END) / max(total), 1) AS top_10pct_share,
           max(claim_amount) AS max_claim
    FROM ranked
""")

run("6. Percentili degli importi dei sinistri (50%, 90%, 99%, 99.9%)", """
    SELECT percentile_approx(claim_amount, array(0.5, 0.9, 0.99, 0.999)) AS percentiles
    FROM lake.silver.fremtpl2_sev
""")

# ---- Piani di esecuzione: tabella partizionata (gold) vs non partizionata (silver) ----
q_gold = f"SELECT sum(claim_nb) / sum(exposure) AS freq FROM {G} WHERE area = 'A'"
q_silver = "SELECT sum(claim_nb) / sum(exposure) AS freq FROM lake.silver.fremtpl2_freq WHERE area = 'A'"

print("\n== 7. Piano di esecuzione, tabella gold (partizionata per area)")
spark.sql(q_gold).explain("formatted")
print("\n== 7b. Piano di esecuzione, tabella silver (non partizionata)")
spark.sql(q_silver).explain("formatted")

run("7c. File Iceberg della gold: totali e quelli della partizione 'A'", f"""
    SELECT count(*) AS total_files,
           sum(CASE WHEN partition.area = 'A' THEN 1 ELSE 0 END) AS files_area_a
    FROM {G}.files
""")


def timed(label, sql, runs=3):
    spark.sql(sql).collect()  # warm-up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        spark.sql(sql).collect()
        times.append(time.perf_counter() - t0)
    print(f"{label}: min={min(times):.3f}s  media={sum(times) / len(times):.3f}s")


print("\n== 8. Tempi (3 esecuzioni dopo warm-up)")
timed("gold partizionata  ", q_gold)
timed("silver non partiz. ", q_silver)
