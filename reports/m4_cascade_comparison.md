# Milestone 4: confronto della cascata (TF-IDF, transformer, combinazione)

Modello: sentence-transformers/all-MiniLM-L6-v2 | lunghezza massima 256 token | addestramento su 9,000 esempi (3,000 positivi), 2 epoch, 59.6 min | inferenza 17.01 testi/s su 4 thread
Test: 138,202 reclami, 3,533 esiti gravi (2.56%). Candidati dello stadio 1: 23.7% del test. Campione pesato: 10,758 righe (peso totale 32,799), bootstrap B=200.
Combinazione: logit = -0.66 +0.44*logit(TF-IDF) +0.51*logit(transformer)

## 1. PR-AUC pesata tra i candidati (la regione in cui i due stadi competono)

```text
                            ranker  pr_auc_candidati  ic_low  ic_high  diff_vs_tfidf  diff_ic_low  diff_ic_high
                       Solo TF-IDF            0.7554   0.737   0.7723              0            0             0
                  Solo transformer            0.7849  0.7671   0.8015        0.02951      0.01681       0.04122
Combinazione (LR sulla validation)            0.7964  0.7805   0.8129        0.04106      0.03505       0.04836
```

## 2. Rivedendo la quota indicata di TUTTI i reclami del test: casi gravi catturati (recall) e precisione

Solo TF-IDF:
```text
 review_rate_%  reviewed  recall  precision  lift
           0.5       691  0.1891     0.9667 37.81
             1      1382  0.3552     0.9081 35.52
             2      2764   0.593     0.7579 29.65
             5      6910  0.8353     0.4271 16.71
            10     13820  0.9137     0.2336 9.137
            15     20730  0.9442     0.1609 6.295
            20     27640  0.9612     0.1229 4.806
```
Solo transformer:
```text
 review_rate_%  reviewed  recall  precision  lift
           0.5       691  0.1902     0.9725 38.04
             1      1382  0.3643     0.9312 36.43
             2      2764  0.6139     0.7847  30.7
             5      6910   0.848     0.4336 16.96
            10     13820  0.9213     0.2355 9.213
            15     20730  0.9454     0.1611 6.302
            20     27640  0.9581     0.1225 4.791
```
Combinazione (LR sulla validation):
```text
 review_rate_%  reviewed  recall  precision  lift
           0.5       691  0.1913     0.9783 38.27
             1      1382   0.362     0.9255  36.2
             2      2764   0.621     0.7938 31.05
             5      6910  0.8681     0.4438 17.36
            10     13820  0.9281     0.2373 9.281
            15     20730  0.9516     0.1622 6.344
            20     27640  0.9621      0.123  4.81
```

## 3. Differenza di recall rispetto al solo TF-IDF, con IC bootstrap al 95% (positivo = meglio del TF-IDF)

```text
 review_rate_%                             ranker  delta_recall    ic_low  ic_high
             1                   Solo transformer      0.009057 -0.003121  0.01841
             1 Combinazione (LR sulla validation)      0.006793 -0.001132  0.01557
             2                   Solo transformer       0.02095  0.006227   0.0351
             2 Combinazione (LR sulla validation)       0.02802   0.01499  0.03879
             5                   Solo transformer       0.01274  0.002809  0.02349
             5 Combinazione (LR sulla validation)       0.03283   0.02432  0.04132
            10                   Solo transformer      0.007642  0.000276  0.01531
            10 Combinazione (LR sulla validation)       0.01444  0.008774  0.02011
```

## 4. Robustezza: per anno del test e per stile della narrativa ('the contact' = scritta da un operatore)
Nel train un prefisso 'TL*' marcava parte dei reclami di operatore; dal 2022 non esiste piu', ma lo stile e' rimasto.
Differenze grandi tra gruppi indicano dove il modello e' meno affidabile e cosa monitorare.

```text
            gruppo  reclami  tasso_grave                             ranker  pr_auc_candidati  recall_5%
        anno: 2025    72051       0.0224                        Solo TF-IDF            0.7638     0.8575
        anno: 2025    72051       0.0224                   Solo transformer            0.8093     0.8674
        anno: 2025    72051       0.0224 Combinazione (LR sulla validation)            0.8121     0.8829
        anno: 2026    66151      0.02901                        Solo TF-IDF            0.7485     0.8192
        anno: 2026    66151      0.02901                   Solo transformer            0.7647     0.8317
        anno: 2026    66151      0.02901 Combinazione (LR sulla validation)            0.7835     0.8447
      stile: altro   109599      0.02264                        Solo TF-IDF            0.6385     0.7868
      stile: altro   109599      0.02264                   Solo transformer            0.6612     0.7981
      stile: altro   109599      0.02264 Combinazione (LR sulla validation)            0.6829     0.8243
stile: the contact    28603      0.03678                        Solo TF-IDF             0.943     0.9487
stile: the contact    28603      0.03678                   Solo transformer            0.9627     0.9667
stile: the contact    28603      0.03678 Combinazione (LR sulla validation)            0.9632     0.9715
```

## 5. Costo atteso per 1.000 reclami (ipotesi: rileggere un reclamo costa 1; non vedere un caso grave costa V)

```text
  V                             ranker  quota_ottima_%  recall  costo  senza_triage  rivedere_tutto  risparmio_%
 10                        Solo TF-IDF             4.5  0.8166  91.89         255.6            1000        64.06
 10                   Solo transformer               4  0.8208   85.8         255.6            1000        66.44
 10 Combinazione (LR sulla validation)               4   0.833  82.69         255.6            1000        67.65
 30                        Solo TF-IDF             7.5  0.8907  158.8         766.9            1000         79.3
 30                   Solo transformer               8  0.9009    156         766.9            1000        79.66
 30 Combinazione (LR sulla validation)             7.5  0.9094  144.5         766.9            1000        81.16
100                        Solo TF-IDF            14.5  0.9428  291.2          2556            1000        70.88
100                   Solo transformer              14  0.9423  287.6          2556            1000        71.24
100 Combinazione (LR sulla validation)              13  0.9474  264.6          2556            1000        73.54
```
I costi sono ipotesi, non misure: servono a mostrare come la scelta della quota di revisione dipende dal rapporto tra il costo di un caso mancato e quello di una rilettura. Da sostituire con i valori dell'azienda.

Costo di calcolo: stadio 2 a 17.01 testi/s -> 0.2 min per 1.000 reclami (solo candidati) contro 1.0 min se il transformer rileggesse tutto; il TF-IDF su tutti i reclami e' trascurabile.
