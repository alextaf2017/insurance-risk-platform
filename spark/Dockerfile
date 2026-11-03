FROM apache/spark:3.5.3-python3

USER root
ARG ICEBERG_VERSION=1.6.1
ARG PG_JDBC_VERSION=42.7.4

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# JAR: runtime Iceberg per Spark 3.5, bundle AWS (S3FileIO), driver JDBC Postgres
RUN set -eux; cd /opt/spark/jars; \
    M=https://repo1.maven.org/maven2; \
    curl -fsSLO $M/org/apache/iceberg/iceberg-spark-runtime-3.5_2.12/${ICEBERG_VERSION}/iceberg-spark-runtime-3.5_2.12-${ICEBERG_VERSION}.jar; \
    curl -fsSLO $M/org/apache/iceberg/iceberg-aws-bundle/${ICEBERG_VERSION}/iceberg-aws-bundle-${ICEBERG_VERSION}.jar; \
    curl -fsSLO $M/org/postgresql/postgresql/${PG_JDBC_VERSION}/postgresql-${PG_JDBC_VERSION}.jar

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
# Container di sviluppo eseguito come root per semplicita con i bind mount.
