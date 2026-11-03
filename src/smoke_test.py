"""Verifica che Spark, Iceberg, Postgres (catalogo) e MinIO comunichino correttamente."""
from common.spark_session import CATALOG, ensure_namespaces, get_spark

spark = get_spark("smoke-test")
ensure_namespaces(spark)

t = f"{CATALOG}.sandbox.smoke"
spark.sql(f"DROP TABLE IF EXISTS {t} PURGE")
spark.sql(f"CREATE TABLE {t} (id INT, msg STRING) USING iceberg")
spark.sql(f"INSERT INTO {t} VALUES (1, 'hello'), (2, 'iceberg')")
spark.sql(f"SELECT * FROM {t}").show()
spark.sql(f"SELECT file_path FROM {t}.files").show(truncate=False)
spark.sql(f"DROP TABLE {t} PURGE")
print("SMOKE TEST OK")
