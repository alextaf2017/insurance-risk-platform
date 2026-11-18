# Milestone 5B: drift dei testi NHTSA anno per anno

Stadio 1 (TF-IDF): soglia fissa dei candidati 0.00928 (top 20% sulla validation). Punteggi fuori campione per gli anni di train (2018-2023), del modello addestrato sul train per validation (2024) e test (2025-2026). Il 2026 e' un anno parziale. Riferimento del PSI: tutto il train. 'Testi lunghi' = oltre 190 parole (circa 256 token).

## 1. Indicatori per anno

```text
 anno  reclami  tasso_grave  parole_medie  testi_lunghi_%  the_contact_%  prima_persona_%
 2018    64447      0.03409         90.83           8.793          17.63            14.21
 2019    74053      0.02997         92.12            9.11          17.46            15.72
 2020    57729      0.02832         90.89           8.815          14.64            17.75
 2021    48353      0.02949         97.89           10.37          19.71             16.1
 2022    52529      0.02776         103.1           11.24          23.46            14.77
 2023    61497      0.02792         102.7           11.19          23.48            14.63
 2024    64962      0.02084           105           11.58          26.26            14.37
 2025    72051       0.0224         110.4           14.53          22.96            15.82
 2026    66151      0.02901         121.6           19.99          18.23            18.87
```

## 2. Stabilita' del punteggio e delle prestazioni

```text
 anno  psi_punteggio_vs_train  psi_punteggio_vs_prec  psi_parole_vs_train  psi_parole_vs_prec  candidati_%  gravi_nei_candidati_%  pr_auc_tfidf  recall_5%  prob_media_prevista  prev_su_oss
 2018                0.002464                    NaN             0.008143                 NaN        25.01                  96.95        0.8121     0.8421              0.03243       0.9514
 2019                0.006704               0.005229             0.004644            0.003121        23.18                  95.49         0.792     0.8427              0.02847       0.9501
 2020                0.004968              0.0002626             0.009291            0.003926        23.03                  95.11        0.7716     0.8379              0.02733       0.9651
 2021               0.0001974               0.005114             0.001206             0.01228        22.75                  96.07        0.7891     0.8457              0.02856       0.9686
 2022                0.005104               0.005228              0.01372            0.008735        23.18                   95.2        0.7716     0.8436              0.02756        0.993
 2023                 0.01346               0.003086              0.01253           0.0003608         22.5                  95.98        0.7409     0.8165              0.02618       0.9378
 2024                 0.05178                0.01556               0.0236            0.002111           20                  95.94        0.7403     0.8634              0.02213        1.062
 2025                 0.02325               0.008474              0.04772             0.01326        21.21                  96.16        0.7371     0.8587              0.02381        1.063
 2026                0.007937                0.03126               0.1189             0.02475        26.48                  96.87        0.7319     0.8197               0.0311        1.072
```
`prev_su_oss` = probabilita' media prevista / tasso grave osservato (1 = nessuna deriva del livello); `candidati_%` usa la soglia fissa.

## 3. Le soglie di riaddestramento proposte nella model card (recall al 5% di riferimento: 0.863, anno 2024)

```text
 anno  candidati_%  scatta_candidati  parole_medie  scatta_lunghezza  recall_5%  scatta_recall  psi>0.10
 2018        25.01              True         90.83             False     0.8421          False     False
 2019        23.18             False         92.12             False     0.8427          False     False
 2020        23.03             False         90.89             False     0.8379          False     False
 2021        22.75             False         97.89             False     0.8457          False     False
 2022        23.18             False         103.1             False     0.8436          False     False
 2023         22.5             False         102.7             False     0.8165          False     False
 2024           20             False           105             False     0.8634          False     False
 2025        21.21             False         110.4             False     0.8587          False     False
 2026        26.48              True         121.6             False     0.8197          False      True
```
Anni in cui scatterebbe almeno una soglia: 2018, 2026. Anni con PSI > 0,10: 2026.
Lettura: una soglia utile scatta prima che le prestazioni peggiorino; se il PSI e' alto ma le prestazioni non cambiano, la soglia e' troppo sensibile; se le prestazioni calano senza che nessuna soglia scatti, ne manca una.
