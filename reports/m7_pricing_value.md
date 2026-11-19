# Milestone 7A: valore economico del prezzo (segmentazione e passaggio da GLM a boosting)

Test: 135,001 polizze, 71,303 anni-polizza, 5,331 sinistri pagati e 1,904 contati senza importo. Costo per sinistro: importo reale con cap a 33,582. Loss ratio obiettivo 70%, spese 25%. Bootstrap B=200 sulle polizze.
I guadagni sono in % del premio del caso base (stesso modello per entrambi): cio' che conta e' l'ordine di grandezza e se l'IC esclude lo zero. I costi sono quelli osservati nel test: con circa 5.000 sinistri il rumore e' grande e i risultati piccoli non sono distinguibili da zero.

## 1. HGB contro GLM: effetto sul risultato tecnico (% del premio del caso base)

`perdente` = il modello peggiore (costo di restare indietro); `migliore` = il modello migliore (guadagno di volume e margine).
```text
 senza_importo mercato  perdente_%  perdente_ic  migliore_%  migliore_ic  quota_migliore  LR_perdente
             0   bassa     -0.5304 -0.64..-0.44    -0.03253 -0.12..+0.08          0.5037       0.7102
             0   media      -2.008 -2.44..-1.63     -0.1635 -0.55..+0.26           0.517       0.7397
             0    alta      -3.681 -4.58..-2.89     -0.3629 -1.16..+0.52          0.5351       0.7758
           0.5   bassa     -0.4641 -0.56..-0.39    -0.09882 -0.17..-0.00          0.5037       0.7089
           0.5   media       -1.76 -2.15..-1.43     -0.4113 -0.74..-0.02           0.517       0.7346
           0.5    alta      -3.228 -4.00..-2.54     -0.8167 -1.50..-0.07          0.5351       0.7659
             1   bassa      -0.415 -0.50..-0.35     -0.1479 -0.22..-0.06          0.5037       0.7079
             1   media      -1.577 -1.95..-1.28     -0.5947 -0.91..-0.22           0.517       0.7307
             1    alta      -2.892 -3.62..-2.30      -1.153 -1.75..-0.42          0.5351       0.7586
```

## 2. GLM contro prezzo piatto: effetto sul risultato tecnico (% del premio del caso base)

`perdente` = il modello peggiore (costo di restare indietro); `migliore` = il modello migliore (guadagno di volume e margine).
```text
 senza_importo mercato  perdente_%  perdente_ic  migliore_%  migliore_ic  quota_migliore  LR_perdente
             0   bassa      -1.459 -1.62..-1.31     0.08283 -0.07..+0.25           0.513       0.7286
             0   media      -5.002 -5.56..-4.49      0.3337 -0.18..+0.92          0.5495       0.8055
             0    alta      -8.726 -9.67..-7.81      0.6566 -0.24..+1.66            0.59       0.9018
           0.5   bassa      -1.194 -1.34..-1.06     -0.1819 -0.32..-0.03           0.513       0.7232
           0.5   media      -4.091 -4.60..-3.66     -0.5771 -1.01..-0.05          0.5495       0.7853
           0.5    alta      -7.147 -8.02..-6.36     -0.9221 -1.72..-0.01            0.59       0.8633
             1   bassa     -0.9985 -1.14..-0.87     -0.3778 -0.50..-0.24           0.513       0.7192
             1   media      -3.417 -3.88..-2.99      -1.251 -1.66..-0.78          0.5495       0.7704
             1    alta      -5.979 -6.77..-5.23       -2.09 -2.82..-1.30            0.59       0.8348
```

## 3. HGB contro prezzo piatto: effetto sul risultato tecnico (% del premio del caso base)

`perdente` = il modello peggiore (costo di restare indietro); `migliore` = il modello migliore (guadagno di volume e margine).
```text
 senza_importo mercato  perdente_%   perdente_ic  migliore_%  migliore_ic  quota_migliore  LR_perdente
             0   bassa      -1.607  -1.77..-1.45     0.06296 -0.09..+0.24          0.5149       0.7316
             0   media      -5.371  -5.91..-4.81      0.2388 -0.31..+0.78          0.5543       0.8144
             0    alta      -9.317 -10.30..-8.38      0.4923 -0.43..+1.46          0.5967        0.919
           0.5   bassa      -1.316  -1.46..-1.17     -0.2284 -0.37..-0.08          0.5149       0.7256
           0.5   media      -4.377  -4.85..-3.91     -0.7549 -1.21..-0.27          0.5543       0.7921
           0.5    alta      -7.588  -8.43..-6.79      -1.236 -1.99..-0.41          0.5967       0.8762
             1   bassa        -1.1  -1.23..-0.97      -0.444 -0.56..-0.32          0.5149       0.7211
             1   media      -3.641  -4.06..-3.25       -1.49 -1.88..-1.09          0.5543       0.7756
             1    alta      -6.309  -7.06..-5.65      -2.515 -3.17..-1.78          0.5967       0.8444
```

## Lettura
- HGB contro GLM, mercato medio e sinistri senza importo al 50%: chi resta con il GLM perde 1.76% del premio (IC -2.15..-1.43); chi ha il boosting ottiene -0.41% (IC -0.74..-0.02).
- Segmentare con un GLM contro non segmentare: chi non segmenta perde 4.09% del premio nello stesso scenario, 2.3 volte il costo di restare col GLM invece del boosting: la segmentazione di base vale molto piu' del passaggio da GLM a boosting.
- Un modello preciso non guadagna molto oltre il margine obiettivo: il suo valore e' soprattutto difensivo, perche' evita le polizze sottoprezzate dal concorrente.
- I sinistri senza importo sono una fonte di incertezza: se costano e sono concentrati in alcuni segmenti, un modello sul solo target dei sinistri pagati li sottoprezza (vedi la dipendenza dal parametro nelle tabelle).
