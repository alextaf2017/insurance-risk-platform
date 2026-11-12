[English version](README.en.md)

# Insurance Claims Risk Platform

Progetto di data engineering e risk analytics per l'assicurazione RC Auto, sviluppato per milestone. **La milestone 1 è completata**: data lake Bronze/Silver/Gold con Apache Spark e Apache Iceberg, controlli di qualità, data contract e split train/test deterministico.

I modelli attuariali, il machine learning, la componente NLP, il deployment e il business case sono **attività pianificate**, non funzionalità già completate.

| Milestone | Contenuto | Stato |
|---|---|---|
| 1 | Data lake (bronze/silver/gold), Iceberg, data quality, data contract | ✅ completata |
| 2 | EDA con Spark SQL, inferenza statistica (GLM, VIF, bootstrap) | ✅ completata |
| 3 | Modelli tabellari: frequenza (GLM/HGB), severità, premio puro, Tweedie | ✅ completata |
| 4 | NLP: triage di narrative NHTSA (TF-IDF contro DistilBERT con fine-tuning) | ⏳ in corso |
| 5 | Risk & responsible AI (bias, drift, overfitting, model card) | ⬜ |
| 6 | Deploy (FastAPI, Docker, CI/CD, monitoraggio) | ⬜ |
| 7 | Business case e ROI | ⬜ |

## Architettura (milestone 1)

```
OpenML (freMTPL2)
      |
 data/raw/*.parquet
      |
 [02_bronze]  grezzo + metadati di ingestion        lake.bronze.*
      |
 [03_silver]  tipi, dedup, regole, quarantena       lake.silver.*
      |
 [04_gold]    data product `policy_features`        lake.gold.*   (partizionata per area)

 Storage:  S3-compatibile (S3Mock in locale)  <-- s3://warehouse/
 Catalogo: Iceberg JDBC su Postgres
 Calcolo:  Spark 3.5 (local mode in container)
```

**Dataset**: freMTPL2 (French Motor Third-Party Liability): `freq` con 678.013 polizze e `sev` con 26.639 sinistri.

## Avvio rapido

Requisiti: Docker (con integrazione WSL 2 su Windows) e `make`. Lavora nel filesystem Linux di WSL
(`~/...`) e non in `/mnt/c/...`, che è molto più lento con i bind mount.

```bash
make up         # build immagine Spark + avvio S3Mock, Postgres, Spark
make smoke      # verifica la catena Spark -> Iceberg -> Postgres -> S3
make pipeline   # download -> bronze -> silver -> gold
make explore    # tabelle, snapshot, file fisici, anomalie
make demo       # time travel, schema evolution, rollback, compattazione
make down       # ferma i container (i dati restano)
make clean      # ferma e cancella i volumi (il lake va perso)
```

Spark UI: http://localhost:4040 mentre un job è in esecuzione.

## Struttura

```
docker-compose.yml        stack locale (S3Mock, Postgres, Spark)
spark/                    Dockerfile (JAR Iceberg, AWS, JDBC) e requirements Python
src/common/               SparkSession con catalogo Iceberg, controlli di data quality
src/ingestion/            01 download, 02 bronze, 03 silver, 04 gold, 05 demo Iceberg
src/explore_lake.py       esplorazione del lake
contracts/                data contract dei data product
```

## Risultati della milestone 1

| Metrica | Valore |
|---|---|
| Polizze in gold | 678.013 |
| Righe scartate in quarantena | 0 |
| Polizze con `exposure` limitata (cap a 1) | 1.224 |
| Polizze con `claim_nb` limitato (cap a 4) | 9 |
| Split train / test (hash di `id_pol`) | 543.012 / 135.001 (≈ 80/20) |
| Frequenza sinistri train / test | 0,1004 / 0,1014 |
| Polizze incoerenti (frequenza vs severità) | 9.116 (≈ 1,3%) |

**Anomalia nota.** Tutte le 9.116 incoerenze sono dello stesso tipo: `claim_nb > 0` ma nessun record
nella tabella di severità. Per queste polizze `avg_severity` è nullo. Le cause (sinistri senza costo,
non ancora liquidati, errori di abbinamento) vanno verificate e la scelta di trattamento (escluderle
dal modello di severità, imputare, analisi di sensibilità) è oggetto della milestone 3.
Il flag `is_claim_inconsistent` rende la scelta tracciabile.

## Risultati della milestone 3 (valutazione sul test)

- **Frequenza**: il gradient boosting ottimizzato (Optuna) batte il GLM con spline: devianza di Poisson 0,574 contro
  0,595 (D² 8,4% contro 5,1%, Gini 0,333 contro 0,287). La differenza è +0,021 per anno-polizza, IC bootstrap 95% (+0,018, +0,023).
  Con parametri di default il boosting sovradatta (degrado train-test 4,6%); il tuning lo riduce.
- **Dato anomalo**: nel portafoglio i sinistri contati (`claim_nb`) superano quelli con importo registrato
  (train: 28.826 contro 21.113, rapporto 0,73, stabile nel test). Moltiplicare frequenza e severità sovrastima
  il costo (rapporto perdita reale su prevista 0,64). Non è casuale: la quota di sinistri con importo è 0,23 per
  i veicoli di età 0 e 0,47 per la marca B12, contro circa 0,8 per il resto.
- **Soluzione adottata**: frequenza dei sinistri pagati (`sev_claim_count`) × severità. Sulle perdite con cap
  il rapporto reale/previsto diventa 1,03 (IC 0,96-1,08); il Tweedie diretto dà 1,01. Gini: 0,330 contro 0,327 (differenza non significativa).
- **Severità**: nessun modello batte la costante di almeno l'1% in cross-validation (migliore: +0,46%), e il
  guadagno cambia segno con la soglia del cap. Si usa la severità media con cap più un fattore di coda.
- **Incertezza principale**: il fattore di coda stimato sul train (1,385) è più alto di quello implicito nel test
  (circa 1,18); a seconda del cap oscilla tra 1,21 e 1,49. Sulle perdite senza cap il rapporto è 0,87.

**Da confermare con chi possiede i dati**: cosa significa un sinistro senza importo. Se costa zero (chiuso senza
pagamento), la frequenza dei sinistri pagati è corretta; se è un costo non ancora registrato, il premio dei
segmenti con molti sinistri senza importo (veicoli nuovi, marca B12) è sottostimato.

## Decisioni di progetto

- **Iceberg con catalogo JDBC (Postgres)** invece di un Hadoop catalog: più vicino a un setup reale e
  adatto agli accessi concorrenti. Offre time travel, schema evolution e partizionamento nascosto.
- **Cap con flag invece di scarto**: `exposure` e `claim_nb` anomali vengono limitati come in letteratura
  attuariale, ma il valore originale (`*_raw`) e i flag restano in silver. I record non validi vanno
  in una tabella di quarantena con la motivazione.
- **Split deterministico** da `xxhash64(id_pol)`: stabile tra esecuzioni e senza leakage tra train e test.
- **Data contract** in `contracts/` con owner, grain, SLA e regole di qualità, come primo passo verso un
  approccio data mesh.
- **Storage S3-compatibile**: in produzione si sostituisce con Azure Blob/ADLS o S3 cambiando solo
  `S3_ENDPOINT`, `WAREHOUSE` e le credenziali. Il codice Spark non cambia.

## Problemi incontrati

| Problema | Causa | Soluzione |
|---|---|---|
| `pull access denied` per `minio/mc` e `minio/minio` | Immagini MinIO non più disponibili su Docker Hub e non accessibili su quay.io (401) | Sostituito con Adobe S3Mock (`adobe/s3mock`) |
| S3Mock risponde `500 InternalError` a ogni scrittura | Il container gira come utente non-root e il volume montato su `/data` è di root (causa probabile: risolto con `user: root`) | `user: root` per il servizio `s3` |
| `cd C:\...` non funziona | WSL richiede percorsi Linux (`/mnt/c/...`) | Progetto spostato in `~/` |

## Limiti noti

- S3Mock è pensato per sviluppo e test: non è un object store per carichi reali.
- Spark gira in local mode in un solo container: non si dimostrano scalabilità e shuffle distribuiti.
- Il container di sviluppo gira come root, quindi i file creati in `data/` sul host risultano di root.

## Prossimi passi

1. **Milestone 2**: EDA con Spark SQL (CTE, window function, `EXPLAIN` prima e dopo il partizionamento),
   poi analisi statistica in pandas (test di significatività, VIF, selezione delle feature).
2. Modelli di frequenza e severità con metriche attuariali (devianza di Poisson, Gini, lift).
3. CI/CD completa con test e build delle immagini (oggi la CI esegue solo lint e controllo del compose).
