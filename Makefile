SUBMIT = docker compose exec spark /opt/spark/bin/spark-submit --driver-memory 3g

.PHONY: up down smoke download bronze silver gold demo explore pipeline shell clean
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
