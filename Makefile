SUBMIT = docker compose exec spark /opt/spark/bin/spark-submit --driver-memory 3g

.PHONY: up down smoke download bronze silver gold demo explore eda stats-export stats model-export model-freq model-sev models model-freq-pay model-sev-pay pipeline shell clean
up:
	docker compose up -d --build
down:
	docker compose down
smoke:
	$(SUBMIT) src/smoke_test.py
download:
	docker compose exec spark python3 src/ingestion/01_download_data.py
bronze:
	$(SUBMIT) src/ingestion/02_bronze.py
silver:
	$(SUBMIT) src/ingestion/03_silver.py
gold:
	$(SUBMIT) src/ingestion/04_gold.py
demo:
	$(SUBMIT) src/ingestion/05_iceberg_demo.py
pipeline: download bronze silver gold
shell:
	docker compose exec spark bash
clean:
	docker compose down -v
explore:
	$(SUBMIT) src/explore_lake.py
eda:
	$(SUBMIT) src/analysis/01_eda_sql.py
stats-export:
	$(SUBMIT) src/analysis/00_export_train.py
stats:
	docker compose exec spark python3 src/analysis/02_stats_inference.py
model-export:
	$(SUBMIT) src/modeling/export_data.py
model-freq:
	docker compose exec spark python3 src/modeling/01_frequency.py
model-sev:
	docker compose exec spark python3 src/modeling/02_severity_pricing.py
models: model-export model-freq model-sev
model-freq-pay:
	docker compose exec -e FREQ_TARGET=sev_claim_count spark python3 src/modeling/01_frequency.py
model-sev-pay:
	docker compose exec -e FREQ_TARGET=sev_claim_count spark python3 src/modeling/02_severity_pricing.py
