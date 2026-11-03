# Milestone 2B: inferenza statistica (solo split train)

Polizze train: 543,012 | sinistri con importo (train): 21,113 | bootstrap B=200

## 1. Frequenza: GLM di Poisson con offset log(exposure)

Dispersione di Pearson (phi): 2.552  (1 = Poisson puro; > 1 sovradispersione)
Devianza residua: 173,207 | devianza nulla: 179,233 | parametri: 46

Test drop-one: aumento di devianza togliendo il termine. `pct_null_dev` è l'effect size; con n grande i p-value sono quasi sempre ~0, quindi conta l'ordine di grandezza dell'effetto.
```text
         term  df_term  delta_dev  pct_null_dev     p_chi2        p_F
  bonus_malus        1       3147         1.756          0 8.349e-270
      veh_age        1      969.2         0.541 8.905e-213  1.519e-84
driv_age_band        4      410.4         0.229  1.539e-87  9.895e-34
       region       21      149.9         0.084  1.839e-21  1.986e-05
    veh_brand       10      103.9         0.058  9.159e-18  1.279e-05
    veh_power        1       16.6         0.009  4.738e-05    0.01088
      veh_gas        1         14         0.008  0.0001855     0.0193
         area        5        9.1         0.005      0.105     0.6133
  log_density        1        4.6         0.003    0.03185     0.1791
```

## 2. Multicollinearità

VIF (campione di 100.000 polizze; regola pratica: > 5 attenzione, > 10 problema):
```text
      feature   VIF
bonus_malus_c  1.34
    veh_age_c  1.02
   driv_age_c  1.31
    veh_power  1.01
  log_density 17.29
    area_code 17.27
```

Correlazione di Spearman:
```text
               bonus_malus_c  veh_age_c  driv_age_c  veh_power  log_density  area_code
bonus_malus_c              1      0.086      -0.573     -0.069        0.138      0.136
    veh_age_c          0.086          1      -0.078     -0.003       -0.102     -0.102
   driv_age_c         -0.573     -0.078           1      0.041       -0.048     -0.048
    veh_power         -0.069     -0.003       0.041          1       -0.011     -0.011
  log_density          0.138     -0.102      -0.048     -0.011            1      0.976
    area_code          0.136     -0.102      -0.048     -0.011        0.976          1
```

## 3. Relatività di frequenza con intervalli di confidenza bootstrap al 95%

Per `driv_age_band` (relatività = frequenza del livello / frequenza del portafoglio):
```text
driv_age_band  claims  relativity  ci_low  ci_high  ci_width
        18-24    1893       1.878   1.802    1.959     0.157
        25-34    5140       0.989   0.966    1.015      0.05
        35-49   10210        0.97   0.956    0.985     0.029
        50-64    7916       0.962   0.942     0.98     0.038
          65+    3667       0.949   0.918    0.976     0.058
```

Per `bm_band` (relatività = frequenza del livello / frequenza del portafoglio):
```text
   bm_band  claims  relativity  ci_low  ci_high  ci_width
   1: <=50   14404       0.795   0.787    0.804     0.018
  2: 51-75    7677       1.126   1.103    1.148     0.046
 3: 76-100    5671       1.575   1.541    1.613     0.071
4: 101-125     948       3.682   3.464    3.904     0.439
   5: 126+     126       4.273   3.546    4.946     1.401
```

Per `area` (relatività = frequenza del livello / frequenza del portafoglio):
```text
area  claims  relativity  ci_low  ci_high  ci_width
   A    4062       0.814   0.793    0.839     0.046
   B    3040       0.879    0.85    0.916     0.065
   C    7888        0.94   0.919    0.957     0.038
   D    6681       1.075   1.055    1.101     0.046
   E    6252       1.221   1.191    1.244     0.053
   F     903       1.381    1.29    1.475     0.185
```

## 4. Severità

Quota dell'importo totale detenuta dai sinistri più grandi:
```text
 top_fraction_%  n_claims  share_of_total_%
            0.1        21              23.8
              1       211              39.8
              5      1056              53.4
             10      2111                61
```

Stimatore di Hill dell'indice di coda alpha (alpha < 2: varianza infinita; < 1: anche media infinita). Una stima stabile al variare di k indica una coda di tipo Pareto.
```text
  k  threshold  alpha    se
 50  6.935e+04  1.178 0.167
100  3.519e+04  1.088 0.109
200  1.717e+04  1.032 0.073
400       9964  1.162 0.058
```

Gamma GLM (link log), test F quasi-likelihood. phi senza cap: 39.5; con cap al 99,5° percentile (33,582): 3.5.
Un effetto che compare solo senza cap è probabilmente guidato da pochi sinistri enormi.
```text
         term  pct_null_dev_all   p_F_all  pct_null_dev_cap   p_F_cap
driv_age_band             3.753 2.516e-07             0.109    0.1073
         area             0.796    0.1733              0.21   0.01193
  log_density             0.414    0.0454             0.027    0.1701
    veh_power             0.219    0.1459             0.051   0.05886
      veh_gas             0.036    0.5545             0.001    0.8217
  bonus_malus             0.014     0.713             0.082   0.01675
      veh_age             0.003    0.8721             0.203 0.0001638
```
