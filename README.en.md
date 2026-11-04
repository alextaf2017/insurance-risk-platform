# Insurance Claims Risk Platform

An insurance data engineering and risk analytics project built in milestones. **Milestone 1 is complete**: a Bronze/Silver/Gold data lake using Apache Spark and Apache Iceberg, data-quality checks, data contracts, and a deterministic train/test split.

Actuarial models, machine learning, NLP, deployment, and the business case are **planned work**, not implemented features yet.

| Milestone | Scope | Status |
|---|---|---|
| 1 | Bronze/Silver/Gold lake, Iceberg, data quality, data contracts | Complete |
| 2 | EDA, statistical inference, optimized SQL | Next |
| 3 | Tabular models (GLM, Random Forest, gradient boosting) | Planned |
| 4 | NLP: transformer fine-tuning on claims text | Planned |
| 5 | Risk & responsible AI (bias, drift, overfitting, model card) | Planned |
| 6 | Deployment (FastAPI, Docker, CI/CD, monitoring) | Planned |
| 7 | Business case and ROI | Planned |

## Architecture — milestone 1

~~~mermaid
flowchart TD
    A[OpenML · freMTPL2] --> B[Raw Parquet]
    B --> C[Bronze · ingestion metadata]
    C --> D[Silver · types, dedup, validation, quarantine]
    D --> E[Gold · policy_features]
    E --> F[Deterministic train/test split]
    subgraph Platform
      G[Apache Spark 3.5]
      H[Apache Iceberg · JDBC catalog]
      I[S3-compatible storage · S3Mock]
    end
    C -.-> G
    D -.-> H
    E -.-> I
~~~

**Dataset:** freMTPL2 (French Motor Third-Party Liability), with 678,013 policies (freq) and 26,639 claims (sev).

## Quick start

Requirements: Docker (with WSL 2 integration on Windows) and make. Work in the Linux filesystem inside WSL (~/...), not under /mnt/c/..., which is significantly slower with bind mounts.

~~~bash
make up         # build Spark image and start S3Mock, Postgres, Spark
make smoke      # verify Spark -> Iceberg -> Postgres -> S3
make pipeline   # download -> bronze -> silver -> gold
make explore    # inspect tables, snapshots, physical files, anomalies
make demo       # time travel, schema evolution, rollback, compaction
make down       # stop containers (data is preserved)
make clean      # stop containers and delete volumes (the lake is lost)
~~~

Spark UI: http://localhost:4040 while a job is running.

## Repository structure

~~~text
docker-compose.yml        local stack (S3Mock, Postgres, Spark)
spark/                    Dockerfile (Iceberg, AWS, JDBC JARs) and Python requirements
src/common/               Iceberg SparkSession and data-quality checks
src/ingestion/            download, bronze, silver, gold, Iceberg demo
src/explore_lake.py       lake exploration
contracts/                data contracts for data products
~~~

## Milestone 1 results

| Metric | Result |
|---|---:|
| Policies in Gold | 678,013 |
| Rows rejected to quarantine | 0 |
| Policies with capped exposure (cap at 1) | 1,224 |
| Policies with capped claim_nb (cap at 4) | 9 |
| Deterministic train/test split (hash of id_pol) | 543,012 / 135,001 (~80/20) |
| Claim frequency, train / test | 0.1004 / 0.1014 |
| Inconsistent policies (frequency vs. severity) | 9,116 (~1.3%) |

**Known data issue.** All 9,116 inconsistencies have the same pattern: claim_nb > 0 but no matching row in the severity table. For these policies, avg_severity is null. Possible causes (claims without a cost, not yet settled, or matching errors) still need investigation. The treatment decision (exclude from severity modeling, impute, or run sensitivity analysis) belongs to milestone 3. The is_claim_inconsistent flag keeps the issue traceable.

## Engineering decisions

- **Iceberg with a JDBC catalog (Postgres):** closer to a production-style setup and supports concurrent access, time travel, schema evolution, and hidden partitioning.
- **Cap with flags instead of silently dropping records:** anomalous exposure and claim_nb values are capped; original *_raw values and flags are retained in Silver. Invalid records are routed to quarantine with a reason.
- **Deterministic split** using xxhash64(id_pol): stable across runs and avoids train/test leakage.
- **Data contracts** in contracts/ define owner, grain, SLA, and quality rules as a first step toward a data-mesh approach.
- **S3-compatible storage:** a production object store (Azure Blob/ADLS or S3) can replace the local S3Mock by changing endpoint, warehouse, and credentials.

## Known limitations

- S3Mock is intended for development and testing, not production workloads.
- Spark runs in local mode in a single container; this project does not demonstrate distributed scaling or shuffle performance.
- The development container runs as root, so files created in host-mounted data/ directories may be owned by root.
- Only milestone 1 is implemented. Statistical analysis, actuarial models, machine learning, NLP, deployment, and monitoring remain on the roadmap.

## Roadmap

1. **Milestone 2 — EDA and statistical analysis:** Spark SQL (CTEs, window functions, EXPLAIN before and after partitioning), followed by statistical analysis in pandas (significance tests, VIF, feature selection).
2. **Milestone 3 — Actuarial modeling:** frequency and severity models with actuarial metrics (Poisson deviance, Gini, lift).
3. **Further milestones:** NLP, responsible-AI evaluation, deployment, CI/CD, monitoring, and business case.
