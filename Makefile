SUBMIT = docker compose exec spark /opt/spark/bin/spark-submit --driver-memory 3g

.PHONY: up down smoke download bronze silver gold demo explore eda stats-export stats model-export model-freq model-sev models model-freq-pay model-sev-pay pipeline shell clean nlp-up nlp-download nlp-lake nlp-smoke nlp-baseline nlp-experiments nlp-stage1 nlp-finetune-dry nlp-finetune nlp-finetune-log nlp-compare nlp-control risk-audit risk-nlp-drift serving-export api-test api-up api-down api-logs api-demo api-demo-shift api-reset-log
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
nlp-up:
	docker compose up -d --build nlp
nlp-download:
	docker compose exec spark python3 src/nlp/01_download_nhtsa.py
nlp-lake:
	$(SUBMIT) src/nlp/02_lake_nhtsa.py
nlp-smoke:
	docker compose exec nlp python3 src/nlp/00_nlp_smoke.py
nlp-baseline:
	docker compose exec spark python3 src/nlp/03_baseline_tfidf.py
nlp-experiments:
	docker compose exec spark python3 src/nlp/04_tfidf_experiments.py
nlp-stage1:
	docker compose exec spark python3 src/nlp/05_stage1_scores.py
nlp-finetune-dry:
	docker compose exec -e DRY_RUN=1 nlp python3 src/nlp/06_finetune_stage2.py
nlp-finetune:
	docker compose exec -d nlp sh -c "python3 src/nlp/06_finetune_stage2.py > data/nlp/finetune.log 2>&1"
nlp-finetune-log:
	tail -n 25 -f data/nlp/finetune.log
nlp-compare:
	docker compose exec spark python3 src/nlp/07_compare_cascade.py
nlp-control:
	docker compose exec spark python3 src/nlp/08_control_lexical_stage2.py
risk-audit:
	docker compose exec spark python3 src/risk/01_insurance_audit.py
risk-nlp-drift:
	docker compose exec spark python3 src/risk/02_nlp_drift.py
serving-export:
	docker compose exec spark python3 src/serving/export_assets.py
api-test:
	docker build -q -f api/Dockerfile --target test -t irp-api-test . && docker run --rm irp-api-test
api-up:
	@mkdir -p data/serving_logs
	SKLEARN_VERSION=$$(python3 -c "import json;print(json.load(open('models/frequency_metadata_payment.json'))['sklearn_version'])" 2>/dev/null || true) API_UID=$$(id -u) API_GID=$$(id -g) docker compose up -d --build api
api-down:
	docker compose stop api
api-logs:
	docker compose logs --tail 50 -f api
api-demo:
	docker compose exec spark python3 src/serving/demo_traffic.py --n 800 --triage 600
api-demo-shift:
	docker compose exec spark python3 src/serving/demo_traffic.py --n 800 --shift --shift-share 0.2 --seed 1
api-reset-log:
	: > data/serving_logs/predictions.jsonl
