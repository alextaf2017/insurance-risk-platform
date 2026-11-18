# Model card: frequenza dei sinistri pagati e premio puro (GLM e gradient boosting)

## Sintesi

Modelli per stimare il costo atteso di una polizza RC Auto come **frequenza dei sinistri pagati × severità × fattore di coda**,
con un benchmark Tweedie diretto. Il gradient boosting è il modello candidato, il GLM con spline il riferimento interpretabile.

| | |
|---|---|
| Target di frequenza | sinistri con importo registrato per anno-polizza (`sev_claim_count / exposure`), peso = esposizione |
| GLM | Poisson con spline cubiche (7 nodi) e one-hot, regolarizzazione L2 minima, scelta con cross-validation |
| Gradient boosting | `HistGradientBoostingRegressor` con perdita di Poisson: 37 foglie, tasso 0,094, minimo 53 polizze per foglia, L2 7,6; early stopping |
| Severità | costante (importo medio con cap al 99,5° percentile del train, 33.582) per il fattore di coda 1,385 |
| Stato | prototipo di ricerca su dati pubblici: non validato per uso operativo |

## Uso previsto e non previsto

- **Previsto**: dimostrare una metodologia attuariale con valutazione critica (calibrazione, equità, robustezza, monitoraggio).
- **Non previsto**: tariffazione di polizze reali, diniego o modifica di coperture, valutazioni su singole persone.
  Il dataset è un unico portafoglio francese; il metodo si trasferisce, i coefficienti no.

## Dati

- freMTPL2 (French Motor Third-Party Liability): 678.013 polizze e 26.639 sinistri con importo. Split per hash di `id_pol`:
  543.012 polizze di train e 135.001 di test (71.303 anni-polizza, 5.331 sinistri pagati, frequenza 0,0748). Il dataset non ha date.
- Variabili: età del conducente, Bonus-Malus, età e potenza del veicolo, alimentazione, marca, regione, densità (logaritmo).
  `area` è esclusa perché correlata 0,976 con la densità.
- **Definizione del target**: i sinistri contati (`ClaimNb`) sono circa il 36% più numerosi di quelli con importo (28.826 contro
  21.113 nel train). La quota con importo non è casuale: 0,23 per veicoli di età 0 e 0,47 per la marca B12, contro circa 0,8 per il resto.
  Si è scelto di modellare i sinistri con importo. **È un'assunzione da confermare con chi possiede i dati**: se i sinistri senza
  importo hanno un costo non registrato, il premio dei segmenti con molti di essi è sottostimato.

## Valutazione (test)

| Misura | GLM | Gradient boosting |
|---|---|---|
| Devianza di Poisson per anno-polizza | 0,4616 | **0,4535** |
| Devianza spiegata rispetto al modello costante | 4,7% | **6,4%** |
| Gini | 0,290 | **0,318** |

Differenza di devianza GLM − HGB: +0,0081 per anno-polizza (IC 95% +0,0058, +0,0110): il boosting è meglio in modo non casuale.
Premio puro (frequenza × severità × coda, boosting): perdita reale su prevista 1,03 (IC 0,96-1,08) sulle perdite con cap per sinistro e 0,87 su
quelle senza cap; il Tweedie diretto dà 1,01 e 0,86. Gini 0,330 (frequenza × severità, boosting) contro 0,327 (Tweedie): non distinguibili.
**L'incertezza principale è la coda**: il fattore di coda stimato sul train (1,385) è più alto di quello implicito nel test (circa 1,18) e
oscilla tra 1,21 e 1,49 a seconda della soglia del cap.

## Audit dei rischi (milestone 5)

**Calibrazione ed equità per gruppo.** Su 35 gruppi (età del conducente, area, regione, età del veicolo, Bonus-Malus, marca, alimentazione)
il boosting ha 1 solo gruppo con intervallo che esclude la calibrazione (35-49 anni: osservati/attesi 1,049, IC 1,012-1,093, cioè sottostima del 4,7%);
il GLM ne ha 3 (veicoli di 16+ anni: 1,139; regione R93: 1,093; alimentazione "regular": 1,038; sottostime tra il 4% e il 14%). Per caso se ne attenderebbero circa 1,8.
I giovani (18-24) hanno una frequenza di 0,169 contro 0,075 (rischio circa 2,25 volte superiore) e il modello la riproduce (1,04; IC 0,94-1,17);
i veicoli di età 0 hanno 0,93 (IC 0,82-1,06). Questo mostra calibrazione per gruppo: **non dimostra equità**. Non si dispone del genere, la regione
può fare da proxy di caratteristiche socio-economiche e se far pagare un giovane circa il doppio sia accettabile è una scelta di politica aziendale e normativa.

**Fallimenti operativi.** Se in produzione una variabile arriva sbagliata (sostituita con la mediana del train), il costo in devianza è:
Bonus-Malus **+7,1%** (il Gini scende da 0,318 a 0,155 e osservati/attesi sul portafoglio va a 1,42: sottostima del 30% circa), età del conducente +1,3% (osservati/attesi 0,86),
densità +0,7%, tutte le altre meno dello 0,6%. Il Bonus-Malus è un punto di fallimento critico ed è in parte endogeno (dipende dai sinistri passati).

**Overfitting e dati.** Lo scarto train-test finale del boosting è il 3,6% e il test è piatto per iterazione (minimo alla 55ª, finale uguale). Il test del boosting
migliora ancora con più dati (0,4560 al 50% del train, 0,4534 al 100%) mentre quello del GLM è piatto da un quarto del train in poi (0,4624 → 0,4616).
Con il 5% del train (27.150 polizze) il GLM è ancora migliore (0,4668 contro 0,4712); a 54.000 polizze il boosting è già avanti (0,4623 contro 0,4634).

## Monitoraggio proposto

- **Qualità degli input**: allarme se Bonus-Malus o età del conducente mancano, sono costanti o cambiano distribuzione.
- **PSI delle variabili e variazione della frequenza media prevista.** Il PSI sul punteggio è poco sensibile: un aumento del 6% del premio medio produce un PSI
  del punteggio di 0,011 e un +21% di 0,096, quindi la soglia convenzionale di 0,10 lascerebbe passare spostamenti rilevanti. Si propone (da validare, non derivata dai dati):
  avviso con PSI di variabile ≥ 0,05, azione con ≥ 0,10, e soglia diretta sulla variazione della frequenza media prevista di ±5% rispetto al riferimento.
  Gli scenari simulano solo cambi di composizione, non cambi nelle relazioni.
- **Calibrazione**: osservati/attesi su portafoglio e per i gruppi principali. Per accorgersi di una deriva del 10% servono circa 784 sinistri pagati
  (10.500 anni-polizza); per una del 5% circa 3.100. Nei gruppi piccoli (giovani, Bonus-Malus > 100, veicoli nuovi) servono da 1,9 a 3,5 volte l'esposizione del test per
  una deriva del 10%: si accorpano i periodi.

## Limiti noti

- Calibrazione verificata sui conteggi di sinistri pagati, non sul costo: la severità è costante e il fattore di coda è incerto.
- Dataset senza date: il drift temporale non è osservabile sui dati assicurativi; gli spostamenti sono simulati.
- Una sola esecuzione di addestramento; il bootstrap tratta le polizze come indipendenti.
- Nessun modello di riserva senza Bonus-Malus (da costruire prima di qualunque uso reale).
- Equità non valutata su attributi protetti non disponibili; le regioni possono essere proxy.

## Riproducibilità

Codice in `src/modeling/` e `src/risk/`, risultati in `reports/m3_*` e `reports/m5_*`, parametri in `models/*_metadata.json`.
