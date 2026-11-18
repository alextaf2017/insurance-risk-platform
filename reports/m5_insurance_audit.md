# Milestone 5A: audit dei rischi dei modelli assicurativi

Modelli: GLM e gradient boosting sul tasso di sinistri con importo (sinistri pagati). Test: 135,001 polizze, 71,303 anni-polizza, 5,331 sinistri pagati (frequenza 0.0748). Bootstrap B=200. Coerenza tra previsioni salvate e modelli ricaricati: differenza massima 0.00e+00.

## 1. Calibrazione ed equita' per gruppo (test)

`rapporto` = sinistri osservati / sinistri attesi dal modello: 1 = calibrato, < 1 = il modello sovrastima il rischio del gruppo (il gruppo paga piu' del dovuto), > 1 = lo sottostima (e' sussidiato dagli altri). `sovrapprezzo` = previsto/osservato - 1. '*' = intervallo di confidenza al 95% che esclude 1. Con molti gruppi alcuni '*' sono attesi per caso (circa 1 su 20).

Per eta_conducente:
```text
gruppo  anni_polizza  osservati  freq_osservata  freq_prevista_HGB  rapporto_HGB  ic_low_HGB  ic_high_HGB fuori_IC_HGB  rapporto_GLM fuori_IC_GLM  sovrapprezzo_HGB_%  dev_GLM  dev_HGB
 18-24          2449        413          0.1686             0.1615         1.044      0.9448        1.165                      1.055                           -4.256   0.8719   0.8526
 25-34     1.281e+04        982         0.07663            0.07887        0.9716      0.9068        1.038                     0.9954                            2.926   0.4868   0.4799
 35-49     2.634e+04       1988         0.07549            0.07197         1.049       1.012        1.093            *         1.036                           -4.662   0.4688   0.4624
 50-64     2.021e+04       1385         0.06851            0.06887        0.9948      0.9358        1.056                      1.001                           0.5244   0.4353   0.4245
   65+          9489        563         0.05933            0.06098        0.9729      0.8887         1.05                     0.9846                            2.789   0.3573   0.3516
```

Per area:
```text
gruppo  anni_polizza  osservati  freq_osservata  freq_prevista_HGB  rapporto_HGB  ic_low_HGB  ic_high_HGB fuori_IC_HGB  rapporto_GLM fuori_IC_GLM  sovrapprezzo_HGB_%  dev_GLM  dev_HGB
     A     1.228e+04        655         0.05332            0.05463         0.976      0.9047        1.041                     0.9821                            2.463   0.3517   0.3482
     B          8548        524          0.0613            0.06271        0.9775      0.8862         1.06                     0.9941                            2.307   0.3943    0.387
     C     2.084e+04       1463         0.07022             0.0675          1.04      0.9954        1.097                      1.031                           -3.861   0.4407   0.4339
     D     1.521e+04       1319         0.08671            0.08408         1.031      0.9728        1.095                      1.047                           -3.034   0.5231   0.5094
     E     1.281e+04       1212          0.0946            0.09607        0.9847      0.9332        1.041                     0.9809                             1.55   0.5541   0.5448
     F          1612        158         0.09801             0.0924         1.061      0.9127        1.245                      1.101                           -5.727   0.6078   0.6064
```

Per regione:
```text
gruppo  anni_polizza  osservati  freq_osservata  freq_prevista_HGB  rapporto_HGB  ic_low_HGB  ic_high_HGB fuori_IC_HGB  rapporto_GLM fuori_IC_GLM  sovrapprezzo_HGB_%  dev_GLM  dev_HGB
   R11          6017        496         0.08244            0.08688        0.9489      0.8683        1.027                     0.9543                            5.387   0.5242   0.5172
   R24     2.023e+04       1304         0.06446            0.06301         1.023      0.9736        1.075                      1.024                           -2.243   0.3966   0.3892
   R52          4342        329         0.07577             0.0717         1.057      0.9467         1.19                      1.078                           -5.378   0.4463    0.436
   R53          5544        363         0.06548            0.06822        0.9598      0.8487         1.07                     0.9547                            4.183   0.3891   0.3843
   R72          2827        219         0.07748            0.07435         1.042       0.909        1.184                       1.05                           -4.034   0.4874   0.4835
   R82          9136        850         0.09304            0.09267         1.004      0.9483        1.065                     0.9957                          -0.4009   0.5364   0.5283
   R91          2936        214          0.0729            0.07033         1.036      0.8848        1.217                      1.015                           -3.517   0.5374   0.4995
   R93          7175        638         0.08892            0.08258         1.077       0.999         1.18                      1.093            *              -7.133   0.5454   0.5392
 altre      1.31e+04        918         0.07009            0.07099        0.9874      0.9328        1.051                      1.001                            1.281   0.4481   0.4425
```

Per eta_veicolo:
```text
gruppo  anni_polizza  osservati  freq_osservata  freq_prevista_HGB  rapporto_HGB  ic_low_HGB  ic_high_HGB fuori_IC_HGB  rapporto_GLM fuori_IC_GLM  sovrapprezzo_HGB_%  dev_GLM  dev_HGB
     0          3384        226         0.06678            0.07177        0.9304      0.8178         1.06                     0.9306                            7.476   0.5075   0.5023
   1-2     1.242e+04        933         0.07514            0.07314         1.027      0.9664        1.108                      1.039                           -2.651   0.4702   0.4675
 11-15     1.556e+04       1146         0.07366            0.07303         1.009      0.9542        1.066                      1.015                          -0.8513   0.4494   0.4329
   16+          5951        356         0.05982             0.0541         1.106      0.9975        1.226                      1.139            *              -9.558   0.3966   0.3892
   3-5     1.426e+04       1108         0.07773            0.07488         1.038      0.9712        1.093                       1.05                           -3.657     0.47   0.4657
  6-10     1.974e+04       1562         0.07914            0.08084        0.9789      0.9402        1.029                     0.9677                            2.151   0.4712   0.4631
```

Per bonus_malus:
```text
 gruppo  anni_polizza  osservati  freq_osservata  freq_prevista_HGB  rapporto_HGB  ic_low_HGB  ic_high_HGB fuori_IC_HGB  rapporto_GLM fuori_IC_GLM  sovrapprezzo_HGB_%  dev_GLM  dev_HGB
101-125           634        210          0.3312             0.3331        0.9944      0.8627        1.128                     0.9869                           0.5657    1.024     1.01
   126+         79.42         37          0.4659             0.4697        0.9918      0.6359         1.31                     0.9148                           0.8271    1.394    1.319
  51-75     1.691e+04       1571         0.09288            0.09163         1.014      0.9698        1.064                       1.01                           -1.339   0.5373   0.5203
 76-100          8943       1156          0.1293             0.1298        0.9959       0.942        1.056                      1.008                           0.4068    0.738   0.7231
   <=50     4.473e+04       2357         0.05269            0.05171         1.019      0.9809        1.067                      1.026                           -1.855    0.368   0.3649
```

Per marca:
```text
gruppo  anni_polizza  osservati  freq_osservata  freq_prevista_HGB  rapporto_HGB  ic_low_HGB  ic_high_HGB fuori_IC_HGB  rapporto_GLM fuori_IC_GLM  sovrapprezzo_HGB_%  dev_GLM  dev_HGB
   B12     1.298e+04        819         0.06309            0.06619        0.9531      0.8825        1.041                     0.9762                            4.917   0.4592   0.4545
 altre     5.832e+04       4512         0.07736            0.07567         1.022      0.9965        1.053                      1.022                            -2.19   0.4621   0.4532
```

Per alimentazione:
```text
   gruppo  anni_polizza  osservati  freq_osservata  freq_prevista_HGB  rapporto_HGB  ic_low_HGB  ic_high_HGB fuori_IC_HGB  rapporto_GLM fuori_IC_GLM  sovrapprezzo_HGB_%  dev_GLM  dev_HGB
 'diesel'     3.404e+04       2663         0.07824            0.07902        0.9902      0.9492        1.031                     0.9928                           0.9944   0.4797   0.4726
'regular'     3.727e+04       2668         0.07159            0.06931         1.033      0.9969        1.077                      1.038            *              -3.187    0.445    0.436
```

Gruppi con rapporto fuori dall'IC: HGB 1 su 35, GLM 3 su 35 (per caso se ne attenderebbero circa 1.8).
I cinque gruppi con la calibrazione peggiore (HGB):
```text
  variabile gruppo  anni_polizza  rapporto_HGB fuori_IC_HGB
eta_veicolo    16+          5951         1.106             
    regione    R93          7175         1.077             
eta_veicolo      0          3384        0.9304             
       area      F          1612         1.061             
    regione    R52          4342         1.057             
```

## 2. Interruzione di una variabile: cosa succede se in produzione arriva vuota o sbagliata

Ogni variabile viene sostituita, da sola, con la mediana del train (numeriche) o con un livello sconosciuto (categoriche). `degrado_%` = aumento della devianza di Poisson rispetto ai dati integri; `rapporto` = osservati/attesi sull'intero portafoglio.
```text
modello variabile_interrotta  devianza  degrado_%    gini  rapporto
    GLM            (nessuna)    0.4616          0  0.2897     1.015
    GLM           driv_age_c    0.4676      1.305  0.2705    0.8325
    GLM            veh_age_c    0.4626     0.2295  0.2833    0.9185
    GLM        bonus_malus_c    0.4955      7.357 0.09665     1.541
    GLM            veh_power     0.462    0.08744  0.2875     1.028
    GLM          log_density    0.4628     0.2692  0.2759     1.023
    GLM              veh_gas    0.4637     0.4611  0.2873      1.18
    GLM            veh_brand    0.4633     0.3799  0.2872     1.092
    GLM               region     0.462     0.1076  0.2851    0.9656
    HGB            (nessuna)    0.4535          0  0.3175     1.011
    HGB           driv_age_c    0.4594       1.31  0.2995    0.8616
    HGB            veh_age_c    0.4556     0.4744  0.3125     0.936
    HGB        bonus_malus_c    0.4857      7.114   0.155     1.423
    HGB            veh_power    0.4547     0.2764  0.3145     1.031
    HGB          log_density    0.4565     0.6581  0.3048     1.075
    HGB              veh_gas    0.4541     0.1328  0.3143     1.044
    HGB            veh_brand     0.456     0.5506  0.3129    0.9996
    HGB               region     0.456      0.548  0.3118     1.036
```
Le variabili la cui interruzione costa di piu' (HGB): bonus_malus_c, driv_age_c, log_density. Per queste serve un controllo di qualita' dei dati in ingresso, con allarme se il valore manca o e' costante.

## 3. Overfitting: curva di apprendimento e curva per iterazione

Devianza di Poisson per anno-polizza, addestrando su una frazione del train (parametri fissi, quelli scelti con Optuna):
```text
modello  quota_train_%  polizze  dev_train  dev_test  scarto_%
    GLM              2    10860     0.4662    0.4735     1.548
    HGB              2    10860     0.3865    0.4738      22.6
    GLM              5    27150     0.4458    0.4668     4.708
    HGB              5    27150     0.3625    0.4712     29.99
    GLM             10    54301     0.4576    0.4634     1.266
    HGB             10    54301     0.4158    0.4623     11.19
    GLM             25   135753     0.4582    0.4624    0.9168
    HGB             25   135753     0.4329    0.4588     5.976
    GLM             50   271506     0.4535    0.4619      1.85
    HGB             50   271506     0.4305     0.456      5.92
    GLM            100   543012     0.4537    0.4616     1.738
    HGB            100   543012     0.4359    0.4534     4.002
```
Uno scarto che si restringe al crescere dei dati e un test che continua a migliorare indicano che altri dati aiuterebbero; un test che si appiattisce con uno scarto che resta largo indica che serve piu' regolarizzazione, non piu' dati.

Gradient boosting: 75 iterazioni selezionate dall'early stopping interno; la devianza di test e' minima all'iterazione 55 (0.45345) e alla fine vale 0.45347 (differenza +0.00%). Scarto finale train-test: 3.6%.

## 4. Spostamenti del portafoglio: PSI e errore di premio (per calibrare le soglie di allarme)

Riferimento: PSI tra train e test (lo split e' casuale, quindi ci si attende un valore vicino a zero; e' il pavimento del rumore):
```text
                     variabile       psi
punteggio (frequenza prevista) 0.0001785
                eta_conducente 8.011e-05
                   eta_veicolo 8.333e-05
                   bonus_malus 6.965e-05
                          area 1.233e-05
```

Si ripesa il test per simulare una composizione diversa del portafoglio (shift delle covariate): il modello resta calibrato per costruzione se le relazioni non cambiano, ma il livello medio del premio si sposta. Il PSI sul punteggio dice a quale spostamento corrisponde una certa soglia di allarme:
```text
                    scenario  moltiplicatore  quota_gruppo_%  psi_punteggio  psi_variabile  var_freq_prevista_%  rapporto_oss_prev
        Piu' giovani (18-24)               1           4.407      0.0001785      8.011e-05                    0              1.011
        Piu' giovani (18-24)             1.5           6.467       0.001185       0.003928                1.998              1.012
        Piu' giovani (18-24)               2           8.441         0.0036         0.0146                 3.93              1.013
        Piu' giovani (18-24)               3           12.15         0.0117         0.0491                7.608              1.015
        Piu' giovani (18-24)               5           18.73        0.03646         0.1461                 14.3              1.019
Auto piu' vecchie (11+ anni)               1           27.28      0.0001785      8.333e-05                    0              1.011
Auto piu' vecchie (11+ anni)             1.5           36.01       0.003191        0.03083                -1.09              1.013
Auto piu' vecchie (11+ anni)               2           42.87       0.009684        0.09475               -1.928              1.015
Auto piu' vecchie (11+ anni)               3           52.96        0.02516          0.249                -3.13              1.018
Auto piu' vecchie (11+ anni)               5           65.23        0.05276         0.5425               -4.548              1.021
  Bonus-malus peggiore (>75)               1            17.5      0.0001785      6.965e-05                    0              1.011
  Bonus-malus peggiore (>75)             1.5           24.14        0.01061        0.02293                6.174              1.009
  Bonus-malus peggiore (>75)               2           29.79        0.03368        0.07397                11.61              1.008
  Bonus-malus peggiore (>75)               3           38.89        0.09605         0.2076                20.75              1.006
  Bonus-malus peggiore (>75)               5           51.47         0.2325         0.4905                34.21              1.003
```
Lettura: con PSI < 0,10 il portafoglio e' stabile; le soglie 0,10 e 0,25 sono convenzioni, e la tabella mostra a quale variazione della frequenza prevista e della composizione corrispondono in questo portafoglio.

## 5. Rilevabilita': quanti sinistri servono per accorgersi di una deriva della frequenza

Approssimazione di Poisson: per rilevare uno spostamento relativo d (95% di confidenza, 80% di potenza) servono circa (2,8/d)^2 sinistri nel gruppo osservato. Gli anni-polizza si ricavano dalla frequenza del gruppo. E' un ordine di grandezza, non una garanzia.
```text
deriva             gruppo  freq_osservata  sinistri_necessari  anni_polizza_necessari  volte_esposizione_gruppo_nel_test
    5% intero portafoglio         0.07477                3136               4.194e+04                             0.5883
    5%         18-24 anni          0.1686                3136                1.86e+04                              7.593
    5%  veicoli di eta' 0         0.06678                3136               4.696e+04                              13.88
    5%  Bonus-malus > 100          0.3462                3136                    9058                               12.7
   10% intero portafoglio         0.07477                 784               1.049e+04                             0.1471
   10%         18-24 anni          0.1686                 784                    4649                              1.898
   10%  veicoli di eta' 0         0.06678                 784               1.174e+04                              3.469
   10%  Bonus-malus > 100          0.3462                 784                    2264                              3.174
   15% intero portafoglio         0.07477               348.4                    4661                            0.06536
   15%         18-24 anni          0.1686               348.4                    2066                             0.8437
   15%  veicoli di eta' 0         0.06678               348.4                    5218                              1.542
   15%  Bonus-malus > 100          0.3462               348.4                    1006                              1.411
   20% intero portafoglio         0.07477                 196                    2622                            0.03677
   20%         18-24 anni          0.1686                 196                    1162                             0.4746
   20%  veicoli di eta' 0         0.06678                 196                    2935                             0.8673
   20%  Bonus-malus > 100          0.3462                 196                   566.1                             0.7935
   30% intero portafoglio         0.07477               87.11                    1165                            0.01634
   30%         18-24 anni          0.1686               87.11                   516.6                             0.2109
   30%  veicoli di eta' 0         0.06678               87.11                    1305                             0.3854
   30%  Bonus-malus > 100          0.3462               87.11                   251.6                             0.3527
```
`volte_esposizione_gruppo_nel_test` = anni-polizza necessari divisi per l'esposizione che il gruppo ha nel test (> 1: servirebbe piu' esposizione di quella disponibile). Nei gruppi piccoli (veicoli nuovi, Bonus-malus alto) il monitoraggio per soglie e' lento: conviene accorpare periodi o usare test sequenziali.
