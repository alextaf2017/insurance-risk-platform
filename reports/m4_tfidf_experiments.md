# Milestone 4: esperimenti con il TF-IDF per progettare il fine-tuning

Train 358,608 (10,652 positivi) | validation 64,962 | test 138,202 (3,533 positivi, tasso 2.56%). PR-AUC casuale sul test: 0.026.

## A. Curva di apprendimento del TF-IDF (campioni con il 25% di positivi; ultima riga: train naturale completo)

```text
 esempi  positivi  pr_auc_val  pr_auc_test  ic_low  ic_high  C  vocabolario
   1000       250      0.5682         0.55  0.5314   0.5655 10        12566
   2500       625      0.6056       0.5892  0.5734   0.6048 10        26839
   5000      1250      0.6351       0.6189  0.6022   0.6341 10        47611
  10000      2500      0.6606       0.6451  0.6306   0.6591 10        81489
  20000      5000      0.6862       0.6699  0.6573   0.6841 10        52062
  40000     10000      0.7031       0.6901  0.6766    0.703 10        88831
  80000     10652      0.7199       0.7108  0.6977   0.7232 10       147401
 358608     10652      0.7398       0.7346  0.7222   0.7469  3       300000
```

## B. Troncamento del testo (TF-IDF su 150,000 reclami di train, C=3): quanto segnale si perde

```text
                     variante  parole_conservate_%  pr_auc_val  pr_auc_test  ic_low  ic_high
                 testo intero                  100      0.7206       0.7132  0.7001   0.7257
 prime 96 parole (~128 token)                87.24      0.6519       0.6413  0.6259    0.656
      testa+coda 48+48 parole                87.24      0.6185       0.5968  0.5788   0.6097
prime 192 parole (~256 token)                97.74      0.7143       0.7055  0.6924   0.7187
```

Se 'prime 96 parole' perde molto rispetto al testo intero, la coda delle narrative contiene segnale (spesso e' li' che si dice chi e' rimasto ferito) e conviene la strategia testa+coda o una lunghezza maggiore.
