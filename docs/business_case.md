# Business case (milestone 7)

Due decisioni valutate con dati del progetto e assunzioni dichiarate: **(1)** passare dal GLM al gradient boosting per il prezzo RC Auto;
**(2)** adottare il triage a due stadi dei reclami. Gli esiti misurati vengono da `make biz-pricing` e `make biz-triage`; scenari e VAN da `make biz-case`.
Le assunzioni stanno in `src/business/assumptions.json` e sono tutte da sostituire con i numeri dell'azienda: il portafoglio di 300.000 anni-polizza,
le spese al 25%, il loss ratio obiettivo al 70%, i costi del team e i minuti per revisione **non sono dati**.

## Sintesi

- **Segmentare conta molto più del modello.** Nel mercato simulato medio, chi non segmenta perde il 4,1% del premio; chi resta col GLM invece di passare al boosting perde l'1,8% (IC 1,4-2,2%).
- **Il boosting è soprattutto una difesa.** Chi prezza meglio non guadagna oltre il margine obiettivo: con i sinistri senza importo al 50% del costo il suo risultato è -0,4% (IC da -0,7 a -0,0). Il valore sta nel non ereditare le polizze sottoprezzate.
- **Il VAN del boosting dipende da ciò che fanno i concorrenti.** Scenario base: +1,41 milioni di euro a 3 anni (IC del test 0,61-2,36 milioni); pessimistico -0,38 milioni; ottimistico +11,0 milioni. Se nessun concorrente migliora il VAN è -1,26 milioni.
- **Il triage ripaga ampiamente se non si valuta ciò che si perde.** Rileggendo il 6,5% dei reclami si trova l'89,5% dei casi gravi: 468 ore l'anno invece di 7.205, risparmio di 303.000 euro contro rileggere tutto. Il triage è preferibile finché un caso grave mancato costa meno di circa 1.560-1.790 euro.
- **Il transformer in più rispetto al solo TF-IDF** trova 33 casi gravi l'anno in più (IC 21-45) allo stesso volume di lavoro: conviene se un caso vale più di circa 1.150 euro (1.870 con il limite basso dell'IC).

## Che cosa è misurato e che cosa è assunto

| Misurato (test, dati reali) | Assunto (da sostituire) |
|---|---|
| Costo di prezzare peggio, con IC bootstrap | Quota di clienti che confrontano i prezzi e sensibilità al prezzo |
| Costi reali dei sinistri del test, con cap a 33.582 | Costo dei sinistri senza importo (0, 50, 100% di un sinistro pagato) |
| Recall del triage per tasso di revisione | Costo orario e minuti per revisione |
| Volumi dei reclami 2025 | Costi di sviluppo, validazione e manutenzione |
| Latenze del servizio (6-8 ms il premio, 50 ms il passaggio dallo stadio 2) | Quota del guadagno che si conserva in produzione, durata del vantaggio, mosse dei concorrenti |

## 1. Prezzo: da GLM a gradient boosting

Metodo: ogni polizza del test è un cliente; due assicuratori la prezzano con modelli diversi e ribasano il livello dei prezzi (conta solo la differenza di relatività);
una quota di clienti sceglie il prezzo più basso. Il costo di ogni polizza è quello osservato nel test. Effetto sul risultato tecnico, in % del premio, mercato "medio"
(30% di clienti che confrontano, sensibilità 10); tra parentesi l'IC bootstrap al 95%:

| Costo dei sinistri senza importo | Chi resta col GLM | Chi passa al boosting | GLM contro nessuna segmentazione (chi non segmenta) |
|---|---|---|---|
| 0% | -2,01 (-2,44; -1,63) | -0,16 (-0,55; +0,26) | -5,00 (-5,56; -4,49) |
| 50% | -1,76 (-2,15; -1,43) | -0,41 (-0,74; -0,02) | -4,09 (-4,60; -3,66) |
| 100% | -1,58 (-1,95; -1,28) | -0,59 (-0,91; -0,22) | -3,42 (-3,88; -2,99) |

- Con mercato poco sensibile (bassa) il costo di restare col GLM è circa lo 0,5% del premio, con mercato molto sensibile (alta) circa il 3,2% (sinistri senza importo al 50%).
- **Il guadagno di chi prezza meglio è nullo o negativo quando i sinistri senza importo costano.** Ipotesi (non verificata): il modello è addestrato solo sui sinistri pagati e i sinistri
  senza importo sono concentrati in certi segmenti (veicoli nuovi, marca B12), quindi chi vince clienti in quei segmenti eredita un costo che il prezzo non contiene.
  Conferma o smentita richiedono il significato di quei sinistri dal titolare dei dati.
- Gli IC misurano solo il rumore dei sinistri osservati (circa 5.300 pagati nel test) e non l'incertezza dell'addestramento del modello: sono un'incertezza minima.

### Scenari (portafoglio di 300.000 anni-polizza, VAN a 3 anni, tasso 8%)

| Scenario | Costo sviluppo (anno 0) | Costo annuo | Beneficio anno 1 | VAN | Pareggio (anni-polizza) |
|---|---|---|---|---|---|
| Pessimistico | 175.500 | 76.500 | -5.000 | -380.000 | non raggiunto |
| Base | 117.000 | 51.000 | 843.000 | 1.409.000 | 44.977 |
| Ottimistico | 93.600 | 40.800 | 4.362.000 | 11.043.000 | 5.304 |

Nel base serve un effetto medio dello 0,10% del premio per pareggiare e la stima è dello 0,67%: un margine di circa 6 volte, che però dipende dalle assunzioni.

**Che cosa sposta il risultato** (VAN base, una assunzione alla volta): la probabilità che i concorrenti migliorino (da -1,26 a +4,08 milioni), la sensibilità del mercato al prezzo (0,20-2,71 milioni),
la quota di guadagno conservata in produzione (0,70-2,12 milioni), la durata del vantaggio (1,06-1,92 milioni), il costo dei sinistri senza importo (1,14-1,68 milioni).
I costi del progetto contano poco (1,28-1,46 milioni al variare di -20%/+50%) rispetto ai ricavi.

## 2. Triage dei reclami

Volumi del 2025: 72.051 reclami, 1.614 con esito grave (2,24%). Tasso di revisione del servizio 6,5%.

| Che cosa | Valore |
|---|---|
| Reclami riletti | 4.683 (6,5%) |
| Ore di revisione con il triage / rileggendo tutto | 468 / 7.205 (0,3 / 4,5 FTE) |
| Risparmio contro rileggere tutto | 303.000 euro l'anno |
| Casi gravi trovati | 1.445 su 1.614 (recall 89,5%, IC 87,0-91,8%) |
| Casi gravi non riletti | 169 |
| Casi gravi trovati rileggendo a caso lo stesso volume | 105 |
| Recall a parità di volume: TF-IDF / transformer / combinazione | 87,4% / 88,0% / 89,5% |
| Casi gravi in più della combinazione sul solo TF-IDF | 33 (IC 21-45) |

**Punti di pareggio** (non richiedono di stimare il valore di un caso): il triage è preferibile a rileggere tutto finché un caso mancato costa meno di 1.789 euro (al lordo dei costi di progetto)
o 1.562 euro (al netto di manutenzione e sviluppo ammortizzato). Il transformer ripaga la propria manutenzione e il proprio sviluppo se un caso in più vale più di 1.149 euro; con il limite basso dell'IC, 1.866 euro.
Il calcolo attribuisce al transformer l'intera manutenzione del triage, quindi è prudente contro il transformer: anche il solo TF-IDF ha costi di manutenzione.
Il VAN a 3 anni del triage contro rileggere tutto è 677.000 euro **prima di dare un valore ai casi mancati**; quel valore dipende da regolatore e responsabilità e non è un dato del progetto.

## 3. Raccomandazione e condizioni

1. **Triage: adottarlo.** Il risparmio è molto più grande del costo del progetto e il pareggio sul caso mancato (circa 1.500-1.800 euro) è basso rispetto al valore di un esito grave in molti contesti. Condizione: decidere con chi usa il servizio quale valore ha un caso mancato e tenere sotto controllo la recall (vedi la model card NLP).
2. **Prezzo: segmentare comunque; passare al boosting solo con condizioni.** Il VAN base è positivo, ma il risultato cambia segno se i concorrenti non migliorano. È un investimento difensivo: conviene se il mercato si muove e se i costi di governance restano quelli ipotizzati.
3. **Prima di decidere**: confermare il significato dei sinistri senza importo (cambia il segno del guadagno offensivo), misurare la sensibilità dei clienti al prezzo sul portafoglio reale, e costruire il modello di riserva senza Bonus-Malus (registro dei rischi #3).

## 4. Limiti

- Il mercato a due assicuratori è un modello stilizzato: nessuna reazione dei prezzi, nessun costo di acquisizione, nessuna regolazione, una sola scelta dei clienti.
- Il portafoglio è illustrativo e i costi sono assunti. Il risultato in euro scala con la dimensione del portafoglio; il pareggio (44.977 anni-polizza nel base) è più robusto del VAN.
- Il test ha circa 5.300 sinistri pagati e una coda pesante: i risultati piccoli (sotto l'1% del premio) non si distinguono dal rumore.
- Il beneficio del triage non è monetizzato: manca il valore di un caso grave trovato in tempo.
- I modelli ereditano i limiti delle model card (target dei sinistri con importo, coda, assenza di date).

## 5. Riproducibilità

```bash
make biz-pricing   # 7A: reports/m7_pricing_value.md/.json, figures/m7_pricing_value.png
make biz-triage    # 7B: reports/m7_triage_value.md/.json
make biz-case      # 7C: reports/m7_business_case.md/.json, m7_business_case_scenarios.csv, figures/m7_business_case_tornado.png
```

Funzioni di calcolo (mercato, VAN, pareggio) in `src/business/bizlib.py`, con test in `tests/test_business.py`.
