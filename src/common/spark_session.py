"""Factory unica per la SparkSession con catalogo Iceberg (JDBC/Postgres) su MinIO."""
import os

from pyspark.sql import SparkSession

CATALOG = "lake"
NAMESPACES = ("bronze", "silver", "gold", "sandbox")


def get_spark(app_name: str = "insurance-risk-platform") -> SparkSession:
    host = os.getenv("CATALOG_DB_HOST", "postgres")
    db = os.getenv("CATALOG_DB_NAME", "iceberg_catalog")
    c = f"spark.sql.catalog.{CATALOG}"

    spark = (
        SparkSession.builder.appName(app_name)
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config(c, "org.apache.iceberg.spark.SparkCatalog")
        .config(f"{c}.catalog-impl", "org.apache.iceberg.jdbc.JdbcCatalog")
        .config(f"{c}.uri", f"jdbc:postgresql://{host}:5432/{db}")
        .config(f"{c}.jdbc.user", os.getenv("CATALOG_DB_USER", "iceberg"))
        .config(f"{c}.jdbc.password", os.getenv("CATALOG_DB_PASSWORD", "iceberg"))
        .config(f"{c}.warehouse", os.getenv("WAREHOUSE", "s3://warehouse/"))
        .config(f"{c}.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
        .config(f"{c}.s3.endpoint", os.getenv("S3_ENDPOINT", "http://minio:9000"))
        .config(f"{c}.s3.path-style-access", "true")
        .config(f"{c}.client.region", os.getenv("AWS_REGION", "us-east-1"))
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def ensure_namespaces(spark: SparkSession) -> None:
    for ns in NAMESPACES:
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CATALOG}.{ns}")
