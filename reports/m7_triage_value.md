# Milestone 7B: valore operativo del triage dei reclami

Test: 138,202 reclami e 3,533 esiti gravi. Volume annuo di riferimento (2025): 72,051 reclami, 1,614 esiti gravi (2.24%). Tasso di revisione del servizio: 6.5%. Bootstrap B=200.

## 1. Recall dei casi gravi per tasso di revisione (misurato)

```text
tasso_revisione           0.020  0.050  0.065  0.100
ranker                                              
Casuale (nessun modello)  0.020  0.050  0.065  0.100
Combinazione (servizio)   0.621  0.868  0.895  0.928
TF-IDF (stadio 1)         0.593  0.835  0.874  0.914
Transformer (stadio 2)    0.614  0.848  0.880  0.921
```

## 2. Che cosa cambia in un anno (volumi del 2025)

- Reclami riletti con il triage: 4,683 (6.5%)
- Ore di revisione con il triage / senza (rileggere tutto): 468 / 7,205  (0.3 / 4.5 FTE)
- Costo di revisione con il triage / senza: 21,075 / 324,230 euro
- Risparmio rispetto a rileggere tutto: 303,155 euro
- Casi gravi trovati (combinazione): 1,445 su 1,614 (recall 89.5%, IC 87.0%..91.8%)
- Casi gravi non trovati (rispetto a rileggere tutto): 169
- Casi gravi trovati rileggendo a caso lo stesso volume: 105
- Casi gravi in piu' del transformer rispetto al solo TF-IDF, stesso volume: 33 (IC 21..45)

## 3. Punti di pareggio (non servono assunzioni sul valore di un caso)

- **Triage contro rileggere tutto**: conviene finche' un caso grave mancato costa meno di 1,789 euro (303,155 euro risparmiati su 169 casi non trovati; al lordo di sviluppo e manutenzione, che considera 7C). Oltre quel valore conviene rileggere tutto.
- **Transformer contro solo TF-IDF**: porta 33 casi in piu' l'anno (IC 21..45) allo stesso volume di lavoro. Il costo aggiuntivo e' la manutenzione (25,000 euro l'anno) e lo sviluppo (40,000 euro, su 3 anni 13,333): conviene se un caso in piu' vale piu' di 1,149 euro (con il limite basso dell'IC: 1,866 euro).

Il valore di un caso grave trovato in tempo non e' un dato di questo progetto (dipende da regolatore, responsabilita' e riserve): i punti di pareggio dicono quanto deve valere perche' ogni scelta si ripaghi.
