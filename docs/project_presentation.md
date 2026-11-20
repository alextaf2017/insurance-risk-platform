# Insurance Claims Risk Platform — presentazione del progetto

Repository: https://github.com/alextaf2017/insurance-risk-platform
Autore: Alessio Tafuni (MSc Statistica, LM-83, Bari)

## 1. Pitch (30 secondi)

Ho costruito da solo una piattaforma end-to-end per il rischio assicurativo, su dati pubblici, in sette tappe: data lake (Spark + Iceberg), analisi statistica, modelli di frequenza sinistri (GLM vs gradient boosting), triage NLP dei reclami, controlli di risk/responsible AI, servizio FastAPI/Docker con monitoraggio e un business case economico. L'obiettivo non era il modello più accurato, ma mostrare l'intera catena: dal dato al deploy, fino a quanto vale in euro e dove si rompe.

## 2. Numeri chiave

| Area | Risultato (test hold-out) |
|---|---|
| Dati | freMTPL2: 678.013 polizze, 26.639 sinistri con importo |
| Frequenza | HGB: deviance 0,4535, Gini 0,318, D² 6,4% — GLM: 0,4616, 0,290, 4,7% |
| Dimensione campione | il GLM è migliore solo fino a ~27.150 polizze di training |
| NLP (NHTSA) | recall 0,868 rivedendo il 5% dei reclami; 0,895 al 6,5% |
| Calibrazione per gruppi | 1/35 gruppi fuori IC per HGB, 3/35 per GLM |
| Guasto di una variabile | senza Bonus-Malus: deviance +7,1%, Gini 0,318 → 0,155 |
| Serving | API con validazione (Bonus-Malus obbligatorio, 422 se manca), log senza testo, 24+ test, CI |
| Monitoraggio | PSI per feature (0,05/0,10), variazione tasso ±5%, n minimo 500 |

## 3. Punti per il CV (italiano)

- Progettato e realizzato una piattaforma end-to-end per il rischio assicurativo (Python, Spark, Iceberg, FastAPI, Docker) su 678.013 polizze e 26.639 sinistri, con 7 milestone documentate, test automatici e CI.
- Costruito un data lake Bronze/Silver/Gold su Apache Iceberg (catalogo JDBC Postgres, storage S3) con data contract, controlli di qualità e split train/test deterministico.
- Confrontato GLM di Poisson (spline) e gradient boosting per la frequenza sinistri (Optuna, Lorenz/Gini, IC bootstrap): Gini 0,318 vs 0,290; individuato che il GLM resta competitivo fino a ~27.000 polizze.
- Sviluppato un triage NLP a cascata (TF-IDF + MiniLM fine-tuned) su reclami NHTSA: recall 0,87 rivedendo il 5% dei casi, con split temporale e pulizia dei leakage.
- Eseguito test di responsible AI: calibrazione per gruppi, simulazione di guasto di una feature, drift nel tempo, model card e risk register.
- Messo in produzione un'API FastAPI containerizzata (multi-stage, utente non-root) con monitoraggio PSI e degradazione controllata; costruito un business case con simulazione di mercato e selezione avversa (NPV, payback, sensibilità).

English version (profile/CV):
- Built an end-to-end insurance risk platform (Spark, Iceberg, scikit-learn, Hugging Face, FastAPI, Docker) on 678k policies, with tests and CI.
- Compared Poisson GLM and gradient boosting (Gini 0.318 vs 0.290) and quantified the sample size below which the GLM wins (~27k policies).
- Built a two-stage NLP triage (TF-IDF + fine-tuned MiniLM): 0.87 recall at 5% review rate.
- Added responsible-AI checks, PSI monitoring and an economic evaluation including adverse selection.

## 4. Storia per il colloquio (struttura)

1. **Problema**: prezzare e controllare il rischio con dati reali e imperfetti.
2. **Scelte**: lakehouse con contratti dati; GLM come baseline interpretabile; boosting come sfidante; cascata NLP per contenere i costi di calcolo.
3. **Risultati**: boosting meglio ma di poco in ranking (Gini +0,03); il GLM basta con pochi dati.
4. **Rischi**: dipendenza da Bonus-Malus (Gini dimezzato se manca), drift, calibrazione per gruppi.
5. **Valore**: business case con selezione avversa; l'esito dipende da ipotesi esplicite.
6. **Limiti** (dirli per primi vale credibilità): vedi sezione 6.

## 5. Domande probabili e risposte

**Perché il GLM e non solo il boosting?** Interpretabilità, regolatorio, stabilità. Ho misurato quando il GLM vince (fino a ~27k polizze) e che il boosting guadagna poco in Gini (0,318 vs 0,290) a costo di spiegabilità.

**Come hai valutato i modelli?** Split deterministico, deviance di Poisson, Gini/Lorenz, calibrazione per gruppi con IC bootstrap, curve di apprendimento. Tuning con Optuna solo sul training.

**Cosa succede se una variabile manca?** L'ho simulato: senza Bonus-Malus la deviance sale del 7,1% e il Gini scende da 0,318 a 0,155. Per questo l'API la richiede (422) e non la imputa. Non ho un modello di riserva senza Bonus-Malus: è un limite dichiarato.

**Come monitori il drift?** PSI per feature (soglie 0,05/0,10), variazione del tasso medio (±5% e >3 errori standard), quota di candidati NLP (±5 punti), n minimo 500. Il PSI è poco sensibile ad alcuni spostamenti: l'ho documentato e usato anche il tasso medio.

**Perché una cascata NLP?** Lo stadio 1 (TF-IDF + LR) seleziona il 20% di candidati, lo stadio 2 (MiniLM) li riordina. Meno calcolo, recall 0,868 al 5% di revisione, split temporale per evitare leakage.

**Il business case è affidabile?** È una simulazione con ipotesi esplicite (due assicuratori, scelta logit, selezione avversa, costi osservati). Il segno del guadagno offensivo dipende da come interpretare i sinistri senza importo: lo dico apertamente e mostro gli scenari e la sensibilità (tornado).

**Cosa rifaresti?** Dati con date (freMTPL2 non le ha), più run del transformer per misurare la varianza, un modello senza Bonus-Malus, validazione su un portafoglio reale.

**Che difficoltà hai avuto?** Un errore di misura nel primo bizlib (un pricer perfetto mostrava guadagno ~0): ho riprogettato la metrica confrontando con lo stesso modello di base. Poi problemi pratici: Python 3.8 nel container Spark, permessi dei file del modello, log cumulativi che falsavano il monitoraggio.

## 6. Limiti onesti

- Solo dati pubblici; nessun portafoglio reale né date dei sinistri.
- I sinistri senza importo hanno significato ambiguo e cambiano il segno del guadagno offensivo.
- Carico di coda (1,385, cap 33.582) è una scelta di modello, non osservata.
- Un solo training del transformer; nessun fallback senza Bonus-Malus.
- Il business case è una simulazione, non una previsione.

## 7. Demo (5 minuti)

`make api-demo` (traffico di esempio e monitoraggio), `make api-demo-shift` (con shift di distribuzione) e lettura del report di monitoraggio; poi `docs/business_case.md` per scenari e sensibilità.
