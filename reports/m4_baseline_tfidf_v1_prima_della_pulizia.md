# Milestone 4: baseline lessicale per il triage delle narrative NHTSA

Train 358,622 (esiti gravi 2.97%) | validation 64,968 (2.08%) | test 138,210 (2.56%) | bootstrap B=200

## 1. Controlli di leakage (train)

Tasso di esito grave per quintile di lunghezza (una forte dipendenza sarebbe un segnale spurio):
```text
       len_bin  reclami  tasso_grave   lift
 (4.999, 38.0]    72928      0.01975 0.6647
  (38.0, 67.0]    70955      0.02259 0.7605
  (67.0, 94.0]    72480      0.02176 0.7325
 (94.0, 137.0]    70733      0.03769  1.269
(137.0, 430.0]    71526      0.04707  1.585
```

Ultima parola della narrativa: livelli con piu' associazione all'etichetta (min 300 reclami). Un lift lontano da 1 su una parola 'di forma' (non di contenuto) indica un artefatto da ripulire:
```text
         tok  reclami  tasso_grave   lift
     deploy.      339       0.3894  13.11
         *dt      708       0.2585  8.701
  collision.      322       0.1149  3.868
         *bf      667      0.09595   3.23
      wheel.      325      0.09231  3.107
   incident.      575       0.0887  2.986
         *tt     1026      0.08382  2.822
    unknown.     2751      0.08179  2.753
      crash.      394      0.08122  2.734
unavailable.      624      0.06731  2.266
    200,000.      343      0.06706  2.257
   accident.     1630      0.06687  2.251
       gear.      321            0      0
  addressed.      315            0      0
    concern.      424     0.002358 0.0794
   warranty.     1008     0.002976 0.1002
     problem      667     0.002999 0.1009
        fix.      636     0.003145 0.1059
```

Prime due parole della narrativa:
```text
               tok  reclami  tasso_grave  lift
           we were      569       0.1019 3.431
             i was    10289      0.09651 3.249
       vehicle was      942      0.09236 3.109
           my wife      625       0.0896 3.016
     the contact's      869      0.08861 2.983
             i got      428      0.07009  2.36
   while traveling      461      0.06941 2.337
   "takata recall"      479      0.06054 2.038
             i had     1442      0.05687 1.914
            on the      320      0.05625 1.894
           tl* the    35852      0.05425 1.826
    takata recall:      316       0.0538 1.811
windshield cracked      365            0     0
          low beam      318            0     0
     the dashboard      387            0     0
    the headlights      465            0     0
   service shifter      329            0     0
    the windshield      563            0     0
```

## 2. Scelta di C (PR-AUC su validation)

```text
  C  val_pr_auc
0.3      0.6933
  1      0.7269
  3      0.7403
 10       0.738
```
Scelto C = 3.0.

## 3. Risultati

PR-AUC sul test con IC bootstrap al 95%. Con un tasso di base del 2.56%, una PR-AUC casuale varrebbe circa 0.026.
```text
                       modello  pr_auc_val  pr_auc_test  ic_low  ic_high  roc_auc_test  base_rate_test
        Regola a parole chiave      0.2411       0.2298  0.2178    0.241        0.7687         0.02556
TF-IDF + regressione logistica      0.7403       0.7347  0.7211   0.7458        0.9716         0.02556
```

Differenza di PR-AUC (TF-IDF - regola a parole chiave) sul test: +0.5049 (IC 95%: +0.4915, +0.5191).

## 4. Tasso di revisione: se un operatore rilegge solo la quota indicata dei reclami con punteggio piu' alto

Regola a parole chiave (test):
```text
 review_rate_%  reviewed  recall  precision  lift
             1      1382  0.1928     0.4928 19.28
             2      2764  0.2966     0.3792 14.83
             5      6910  0.4096     0.2094 8.192
            10     13821  0.5652     0.1445 5.652
            20     27642  0.6493    0.08299 3.247
```

TF-IDF + regressione logistica (test):
```text
 review_rate_%  reviewed  recall  precision  lift
             1      1382  0.3544     0.9059 35.44
             2      2764   0.602     0.7695  30.1
             5      6910  0.8353     0.4271 16.71
            10     13821  0.9157     0.2341 9.157
            20     27642  0.9607     0.1228 4.803
```

## 5. Calibrazione del TF-IDF (test, per decili di probabilita')
Brier score: validation 0.0094 | test 0.0117 (riferimento costante: 0.0249).
```text
    n  observed  predicted
13821 0.0001447  0.0004928
13821 0.0001447    0.00101
13821 0.0002894    0.00155
13821  0.001013   0.002188
13821  0.001302   0.003004
13821  0.001375   0.004126
13821  0.002532   0.005824
13821  0.003256   0.008838
13821    0.0115    0.01657
13821    0.2341     0.2286
```

## 6. Parole e bigrammi con il coefficiente piu' alto: se compaiono parole di forma o firme, c'e' un artefatto
```text
piu_associate_al_grave  coef+ piu_associate_al_non_grave  coef-
               medical  15.67                no injuries -13.67
             sustained  11.15                not injured -6.384
              injuries  10.24                    one was -4.914
                injury  9.733                      crack -4.033
              hospital  9.603                        the  -3.95
     medical attention  9.354                 no medical -3.828
               injured  9.272               any injuries -3.806
     contact sustained  9.088                        oil -3.637
              suffered  9.047                  no injury -3.568
                   leg  8.697                      could -3.523
              accident  8.557                      miles -3.498
                  neck   8.28                injuries or -3.436
                    dt  7.673                  injury or -3.361
                  head  7.666                fortunately   -3.3
             attention  7.316                    were no -3.253
                 crash  7.217             were sustained -3.236
            concussion  7.216                      start -3.217
                   cut  6.963                   that the -3.175
                struck  6.944                    problem -3.172
                 burns  6.811                     no one -3.162
          the hospital   6.76                 windshield -3.101
                   hit  6.709                      light -3.068
                finger  6.612                    luckily -3.058
                  pain  6.414                    shaking -2.964
                  skin  6.277                not sustain -2.941
```
