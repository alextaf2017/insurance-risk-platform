# Milestone 7: business case

Portafoglio illustrativo: 300,000 anni-polizza | VAN a 3 anni, tasso 8% | spese 25%, loss ratio obiettivo 70%. **I valori di scenario (mercato, costi, quota di guadagno conservata) sono assunzioni, non dati**: stanno in `src/business/assumptions.json` e vanno sostituiti con quelli dell'azienda. Sono misurati: il costo di essere meno preciso del mercato (7A) e il comportamento del triage (7B).

## A. Passare da GLM a gradient boosting per il prezzo

```text
    scenario  costo_dev_anno0  costo_annuo  beneficio_anno1        VAN  rientro_anno  pareggio_anni_polizza
pessimistico          175,500       76,500           -4,997   -380,409           NaN                    NaN
        base          117,000       51,000          842,922  1,408,622             1                 44,977
 ottimistico           93,600       40,800        4,362,367 11,043,496             1                  5,304
```

Nello scenario base il valore atteso viene per 50% dal costo evitato di restare col GLM mentre i concorrenti migliorano (stima 1.76% del premio) e per il resto dal guadagno di volume (-0.41%). Estremi dell'IC bootstrap del test per lo scenario base: VAN da 605,658 a 2,362,391 euro.
Attenzione: con circa 5.000 sinistri nel test il guadagno HGB contro GLM e' piccolo rispetto al rumore: se l'IC e' largo, e' il segnale piu' importante di questo business case.

**Beneficio minimo per il pareggio** (scenario base): serve un effetto medio di 0.10% del premio (mercato simulato) per VAN zero; la stima del test, pesata sulle probabilita' di scenario, e' 0.67%. Il minimo dipende da costi, dimensione del portafoglio e premio medio, non dall'effetto misurato: se i due valori sono vicini, decidono le assunzioni.

### Sensibilita' del VAN (euro) a una assunzione alla volta, resto sul base

```text
                                parametro         valore        VAN
           sinistri senza importo (costo)            0.0  1,679,003
           sinistri senza importo (costo)            0.5  1,408,622
           sinistri senza importo (costo)            1.0  1,138,241
         mercato (sensibilita' al prezzo)          bassa    200,237
         mercato (sensibilita' al prezzo)          media  1,408,622
         mercato (sensibilita' al prezzo)           alta  2,712,838
                          fattore di coda          basso  1,163,354
                          fattore di coda           base  1,408,622
                          fattore di coda           alto  1,534,247
        guadagno conservato in produzione            0.4    698,456
        guadagno conservato in produzione            0.7  1,408,622
        guadagno conservato in produzione            1.0  2,118,788
    concorrenti migliorano (probabilita')            0.0 -1,258,829
    concorrenti migliorano (probabilita')            0.5  1,408,622
    concorrenti migliorano (probabilita')            1.0  4,076,073
                     durata del vantaggio [1, 0.5, 0.25]  1,060,670
                     durata del vantaggio [1, 0.75, 0.5]  1,408,622
                     durata del vantaggio      [1, 1, 1]  1,923,859
                 moltiplicatore dei costi            0.8  1,458,309
                 moltiplicatore dei costi            1.0  1,408,622
                 moltiplicatore dei costi            1.5  1,284,406
dimensione del portafoglio (anni-polizza)         100000    303,919
dimensione del portafoglio (anni-polizza)         300000  1,408,622
dimensione del portafoglio (anni-polizza)        1000000  5,275,082
```

L'assunzione che sposta di piu' il VAN e' **concorrenti migliorano (probabilita')** (da -1,258,829 a 4,076,073 euro). Punto di pareggio nello scenario base: 44,977 anni-polizza

## B. Triage dei reclami

Con volumi 2025 (72,051 reclami, 1,614 gravi): il triage rilegge il 6.5% dei reclami e trova il 89.5% dei casi gravi (IC 87.0%..91.8%). Rispetto a rileggere tutto risparmia 303,155 euro l'anno (6,737 ore) e lascia 169 casi gravi non riletti.
VAN a 3 anni del triage contro rileggere tutto (costo di sviluppo 40,000, manutenzione 25,000 l'anno): 676,831 euro, **prima di attribuire un costo ai casi mancati**. Il triage e' preferibile a rileggere tutto finche' un caso grave mancato vale meno di 1,562 euro (risparmio annuo al netto di manutenzione e sviluppo ammortizzato, diviso i casi non riletti).
Il transformer in piu' rispetto al solo TF-IDF trova 33 casi in piu' l'anno (IC 21..45): vedi `reports/m7_triage_value.md` per il suo punto di pareggio.

## C. Che cosa dice il business case
- Prezzo, GLM contro boosting: VAN positivo in 2 scenari su 3 (base, ottimistico).
- Costo di non segmentare (4.09% del premio) contro costo di restare col GLM invece del boosting (1.76%), scenario base: la segmentazione di base vale molto piu' del passaggio al boosting.
- Prima di impegnare risorse: confermare con i titolari dei dati il costo dei sinistri senza importo, la coda e la quota di clienti che confrontano i prezzi.
- Il costo di governance (validazione, monitoraggio, riaddestramento) pesa quanto lo sviluppo: il beneficio deve superarlo ogni anno, non solo una volta.
