# Milestone 4: baseline lessicale per il triage delle narrative NHTSA

Train 358,608 (esiti gravi 2.97%) | validation 64,962 (2.08%) | test 138,202 (2.56%) | bootstrap B=200

## 1. Controlli di leakage (train)

Tasso di esito grave per quintile di lunghezza (una forte dipendenza sarebbe un segnale spurio):
```text
       len_bin  reclami  tasso_grave   lift
 (4.999, 38.0]    72945      0.01975 0.6651
  (38.0, 67.0]    71370       0.0225 0.7576
  (67.0, 94.0]    72796      0.02202 0.7413
 (94.0, 137.0]    70128      0.03786  1.275
(137.0, 430.0]    71369       0.0469  1.579
```

Ultima parola della narrativa: livelli con piu' associazione all'etichetta (min 300 reclami). Un lift lontano da 1 su una parola 'di forma' (non di contenuto) indica un artefatto da ripulire:
```text
         tok  reclami  tasso_grave    lift
     deploy.      350       0.3914   13.18
  collision.      326       0.1166   3.924
      wheel.      329      0.09422   3.172
      crash.      403      0.09181   3.091
   incident.      579      0.09154   3.082
    unknown.     2856      0.08683   2.923
   accident.     1659      0.07535   2.537
    200,000.      357      0.07283   2.452
     40,000.      518      0.06757   2.275
unavailable.      636      0.06604   2.223
     street.      335      0.06567   2.211
     30,000.      418      0.06459   2.175
       gear.      321            0       0
  addressed.      316            0       0
    concern.      427     0.002342 0.07884
     problem      668     0.002994  0.1008
        fix.      636     0.003145  0.1059
       year.      568     0.003521  0.1185
```

Prime due parole della narrativa:
```text
               tok  reclami  tasso_grave  lift
           we were      569       0.1019 3.432
             i was    10289      0.09651 3.249
       vehicle was      942      0.09236 3.109
           my wife      625       0.0896 3.016
     the contact's     1090      0.08716 2.934
             i got      428      0.07009  2.36
   while traveling      461      0.06941 2.337
   "takata recall"      479      0.06054 2.038
             i had     1442      0.05687 1.914
            on the      320      0.05625 1.894
    takata recall:      316       0.0538 1.811
           car was      545      0.05321 1.791
windshield cracked      365            0     0
          low beam      318            0     0
     the dashboard      387            0     0
   service shifter      329            0     0
    the windshield      563            0     0
    the headlights      465            0     0
```

## 2. Scelta di C (PR-AUC su validation)

```text
  C  val_pr_auc
0.3      0.6936
  1      0.7269
  3      0.7403
 10      0.7382
```
Scelto C = 3.0.

## 3. Risultati

PR-AUC sul test con IC bootstrap al 95%. Con un tasso di base del 2.56%, una PR-AUC casuale varrebbe circa 0.026.
```text
                       modello  pr_auc_val  pr_auc_test  ic_low  ic_high  roc_auc_test  base_rate_test
        Regola a parole chiave      0.2411       0.2298  0.2174   0.2432        0.7687         0.02556
TF-IDF + regressione logistica      0.7403       0.7343  0.7208   0.7482        0.9715         0.02556
```

Differenza di PR-AUC (TF-IDF - regola a parole chiave) sul test: +0.5044 (IC 95%: +0.4917, +0.5167).

## 4. Tasso di revisione: se un operatore rilegge solo la quota indicata dei reclami con punteggio piu' alto

Regola a parole chiave (test):
```text
 review_rate_%  reviewed  recall  precision  lift
             1      1382  0.1896     0.4848 18.96
             2      2764  0.2949      0.377 14.75
             5      6910  0.4025     0.2058  8.05
            10     13820  0.5599     0.1431 5.599
            20     27640  0.6513    0.08325 3.256
```

TF-IDF + regressione logistica (test):
```text
 review_rate_%  reviewed  recall  precision  lift
             1      1382  0.3541     0.9052 35.41
             2      2764  0.6026     0.7703 30.13
             5      6910  0.8361     0.4275 16.72
            10     13820   0.914     0.2336  9.14
            20     27640  0.9612     0.1229 4.806
```

## 5. Calibrazione del TF-IDF (test, per decili di probabilita')
Brier score: validation 0.0094 | test 0.0117 (riferimento costante: 0.0249).
```text
    n  observed  predicted
13821 0.0001447    0.00048
13820 0.0002171  0.0009978
13820 0.0002171   0.001539
13820  0.001013   0.002179
13820  0.001013   0.002998
13820  0.001664   0.004126
13820  0.002677   0.005829
13820  0.002967   0.008865
13820   0.01208    0.01664
13821    0.2336     0.2293
```

## 6. Parole e bigrammi con il coefficiente piu' alto: se compaiono parole di forma o firme, c'e' un artefatto
```text
piu_associate_al_grave  coef+ piu_associate_al_non_grave  coef-
               medical  15.64                no injuries -13.68
             sustained  11.14                not injured -6.441
              injuries   10.3                    one was  -4.91
                injury  9.771                      crack -4.038
              hospital   9.59                        the -3.958
     medical attention  9.333               any injuries -3.839
               injured  9.307                 no medical  -3.78
     contact sustained  9.052                        oil -3.666
              suffered  8.951                      could -3.608
                   leg  8.712                  no injury -3.545
              accident  8.662                injuries or -3.484
                  neck  8.359                      miles -3.468
                  head  7.649                  injury or -3.393
             attention  7.276                fortunately -3.312
            concussion  7.258                    were no -3.281
                 crash  7.237                   that the -3.256
                   cut  6.965                      start -3.238
                struck  6.906             were sustained -3.212
                 burns  6.743                    problem -3.174
                   hit  6.701                     no one -3.152
          the hospital    6.7                 windshield -3.125
                finger  6.658                      light -3.055
                  pain  6.515                    luckily -3.049
                  skin   6.23                    shaking     -3
                 sharp  6.223                    back on -2.924
```
