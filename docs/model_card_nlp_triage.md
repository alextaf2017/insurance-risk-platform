# Model card: triage di narrative per esito grave (cascata TF-IDF + transformer)

## Sintesi

Sistema a due stadi che, data la narrativa libera di un reclamo, ordina i reclami per probabilità di **esito grave**
(persone ferite, decessi o cure mediche) per decidere quali far rileggere per primi a un operatore.

| | |
|---|---|
| Stadio 1 | TF-IDF (unigrammi e bigrammi) + regressione logistica, su tutti i reclami |
| Stadio 2 | `sentence-transformers/all-MiniLM-L6-v2` (23M parametri) con fine-tuning, sul 20% con il punteggio più alto |
| Combinazione | regressione logistica su logit dei due punteggi, addestrata sulla validation |
| Stato | prototipo di ricerca: non validato per uso operativo |

## Uso previsto e non previsto

- **Previsto**: dimostrare una metodologia di triage su narrative di incidenti stradali; studio del compromesso tra
  qualità, costo di calcolo e complessità.
- **Non previsto**: decisioni automatiche su pratiche reali, diniego o riduzione di coperture, valutazioni su persone.
  I dati sono reclami su difetti dei veicoli presentati a NHTSA, **non sinistri assicurativi**: il metodo si trasferisce,
  il modello no.

## Dati

- Fonte: reclami dei consumatori NHTSA (dominio pubblico), file piatto ufficiale. 561.772 reclami unici dal 2018 al
  5 novembre 2026, dopo filtro sui soli veicoli, aggregazione per `ODINO` e deduplica dei testi identici.
- Minimizzazione: esclusi città, VIN, dati del concessionario e nome dell'operatore del veicolo.
  Il testo libero può contenere dati personali non oscurati: va trattato come sensibile.
- **Etichetta**: `INJURED > 0` oppure `DEATHS > 0` oppure `MEDICAL_ATTN = 'Y'`. È dichiarata da chi presenta il reclamo
  e non verificata; un "no" significa spesso "non indicato" (dal maggio 2021 i campi vuoti compaiono come "N").
  Nella pratica coincide in gran parte con "qualcuno è rimasto ferito" (14.834 reclami con feriti su 15.539 gravi).
- Split temporale per data di ricezione: train fino al 2023 (358.608 reclami, 2,97% gravi), validation 2024
  (64.962, 2,08%), test dal 2025 (138.202, 2,56%, 3.533 gravi).
- Pulizia: oscuramenti FOIA, spazi unicode, sigle dell'analista in coda (anche multiple) ed etichetta di origine in testa
  (`TL*`) rimossi. Misura di leakage: tolti questi artefatti la PR-AUC del baseline non cambia (0,7347 → 0,7343).

## Addestramento

- Stadio 1: C = 3 scelto sulla validation, 195.664 caratteristiche. Candidati = 20% con punteggio più alto sulla validation
  (soglia 0,00928); nel test sono il 23,7% e contengono il **96,5%** dei casi gravi (il resto è irrecuperabile dalla cascata).
- Stadio 2: addestrato sui candidati del train con punteggi dello stadio 1 calcolati fuori campione (5 fold). 9.000 esempi
  (3.000 gravi), lunghezza massima 256 token (12% dei testi troncati), 2 epoch, batch 16, tasso di apprendimento 5e-5.
  Circa 60 minuti su 4 thread di CPU. Una sola esecuzione (nessuna stima della variabilità tra semi).

## Valutazione (test, 2025-2026)

Tutte le stime del confronto usano un campione pesato del test (tutti i gravi tra i candidati, 25% dei non gravi con peso 4);
gli intervalli sono bootstrap al 95% sui reclami.

| Misura | Solo TF-IDF | Solo transformer | Combinazione |
|---|---|---|---|
| PR-AUC tra i candidati | 0,755 (0,737-0,772) | 0,785 (0,767-0,802) | **0,796** (0,781-0,813) |
| Differenza dal TF-IDF | – | +0,030 (+0,017, +0,041) | **+0,041** (+0,035, +0,048) |
| Recall rivedendo il 2% dei reclami | 59,3% | 61,4% | 62,1% |
| Recall rivedendo il 5% | 83,5% | 84,8% | **86,8%** (+3,3 pt: +2,4, +4,1) |
| Recall rivedendo il 10% | 91,4% | 92,1% | 92,8% |

Il TF-IDF da solo, su tutto il test: PR-AUC 0,734 (0,721-0,748), ROC-AUC 0,972. Una regola a parole chiave dà 0,230.

**Controllo**: un secondo stadio *lessicale* addestrato sugli stessi candidati non migliora il TF-IDF (+0,005 di PR-AUC).
Il transformer batte quel secondo stadio lessicale di +0,024 di PR-AUC (+0,012, +0,037); la combinazione con il transformer
batte la combinazione con il lessicale di +0,036 (+0,030, +0,043) di PR-AUC e di +2,9 pt (+2,0, +3,7) di recall al 5%.
Quindi il guadagno non è un effetto della sola specializzazione sui casi difficili.

**Costo** (ipotesi, da sostituire con valori reali): con un caso mancato che costa da 10 a 100 volte una rilettura, la
combinazione riduce il costo atteso di circa il 9-10% rispetto alla cascata con solo TF-IDF. Calcolo: 17 testi al secondo,
circa 14 secondi per 1.000 reclami.

## Limiti e rischi noti

- **Dominio**: reclami NHTSA in inglese, non sinistri assicurativi. Nessuna validazione su testi italiani.
- **Etichette rumorose**: dichiarate dall'utente, "non indicato" confuso con "no". Il modello impara a prevedere cosa viene
  *dichiarato*, non cosa è accaduto.
- **Tetto della cascata**: il 3,5% dei casi gravi sfugge allo stadio 1 e non viene mai riletto.
- **Drift osservato**: dal 2018 al 2026 la lunghezza media passa da 91 a 122 parole (+34%), il prefisso `TL*` sparisce
  nel 2022 mentre lo stile da operatore ("the contact") resta e oscilla (17,6% → 26,3% → 18,2%); il tasso di esito grave
  varia tra il 2,1% e il 3,4%. Testi più lunghi saranno troncati più spesso a 256 token.
- **Eterogeneità tra gruppi**: sul test la PR-AUC tra i candidati va da 0,94-0,96 per le narrative scritte da un operatore
  ("the contact", 21% dei reclami, tasso grave 3,7%) a 0,64-0,68 per tutte le altre (79%, tasso grave 2,3%); il recall rivedendo il
  5% va da 95-97% a 79-82%. Il guadagno del transformer è maggiore dove il lessico è più debole (+0,044 di PR-AUC e +3,8 punti di
  recall contro +0,020 e +2,3). Per anno: nel 2026 tutti i ranker peggiorano (recall al 5% della combinazione da 88,3% a 84,5%),
  ma il tasso di esito grave sale (2,2% → 2,9%) e a revisione fissa il recall dipende anche dalla prevalenza: con questi dati la
  causa non è separabile. Il guadagno della combinazione è invece uguale nei due anni (+2,5 e +2,6 punti).
- **Statistica**: una sola esecuzione di addestramento, un solo modello e un solo insieme di iperparametri; il bootstrap
  tratta i reclami come indipendenti (reclami sullo stesso difetto sono correlati). I vantaggi sono moderati (+3 punti di recall).
- **Equità**: non valutata. Non si hanno attributi sensibili e lo stile di scrittura può correlarsi con lingua madre o
  istruzione; va verificato prima di qualunque uso reale.
- **Privacy**: il testo può contenere dati personali; gli esempi di errore si salvano solo in locale (`data/nlp/`, ignorata da git).

## Monitoraggio e riaddestramento proposti

- Mensile: distribuzione dei punteggi, quota dei candidati (oggi 20-24%), lunghezza media e quota di testi troncati,
  quota di stile "the contact" e di prima persona, tasso di esito grave dichiarato.
- Con etichette ritardate: recall al 5% di revisione su base trimestrale.
- Riaddestrare se: la quota dei candidati si sposta oltre ±5 punti, il recall al 5% scende di oltre 5 punti rispetto
  a 86,8%, o la lunghezza media supera le 140 parole.

## Riproducibilità

Tutto è nel repository: `src/nlp/` (ingestion, baseline, esperimenti, cascata, controllo), `reports/m4_*` (risultati),
`models/nlp_*_metadata.json` (parametri e metriche). I punteggi grezzi e i checkpoint non sono versionati.
