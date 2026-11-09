# Milestone 3A: modelli di frequenza

Train: 543,012 polizze | test: 135,001 polizze | trial Optuna: 25 | bootstrap B=200 | frequenza di riferimento (train): 0.1004

## Risultati sul test (devianza di Poisson per anno-polizza: piu' bassa = meglio)

`gap_%` = degrado da train a test (indizio di overfitting); `D2_test_%` = quota di devianza spiegata rispetto al modello costante; `gini_test` = potere discriminante (0 = nessuno).
```text
                             model  dev_train  dev_test  gap_%  D2_test_%  gini_test
                              Null     0.6244    0.6269 0.4007          0          0
                     GLM (default)     0.5968    0.6002  0.569      4.264     0.2692
                       GLM (tuned)     0.5909    0.5949 0.6814      5.097     0.2869
GLM (area al posto di log_density)      0.591     0.595 0.6825      5.083     0.2868
                     HGB (default)     0.5501    0.5751  4.554      8.257     0.3324
                       HGB (tuned)     0.5563    0.5739  3.162      8.447     0.3333
```

## Parametri scelti da Optuna (CV a 3 gruppi su un sottocampione del train)

- GLM: {"n_knots": 8, "alpha": 1e-06} (CV: 0.59521)
- HGB: {"learning_rate": 0.0722, "max_leaf_nodes": 24, "min_samples_leaf": 94, "l2_regularization": 0.001} (CV: 0.57856)

## GLM (tuned) contro HGB (tuned): differenza di devianza, IC bootstrap al 95% sulle polizze del test

Delta = devianza GLM - devianza HGB = +0.02100 per anno-polizza (IC: +0.01843, +0.02336). Negativo = GLM meglio; positivo = HGB meglio; se l'IC contiene 0 la differenza non e' distinguibile dal rumore.

## Calibrazione per decili di esposizione sul test (HGB (tuned): tasso di sinistri osservato vs previsto)

```text
 decile  exposure  observed  predicted  obs/pred
      1      7130   0.04095    0.03925     1.043
      2      7130   0.05357    0.05264     1.018
      3      7130   0.05723    0.06139    0.9322
      4      7131   0.06746    0.06904    0.9771
      5      7130   0.08191    0.07596     1.078
      6      7131   0.08316    0.08234      1.01
      7      7131   0.08961    0.08963    0.9999
      8      7130   0.09789     0.1019    0.9611
      9      7131    0.1227     0.1328     0.924
     10      7131    0.3195     0.3101      1.03
```

## Importanza per permutazione (HGB tuned, 60.000 polizze del test): aumento di devianza se si mescola la variabile

```text
      feature  dev_increase       std
bonus_malus_c       0.04005  0.001291
    veh_age_c       0.03954  0.001741
    veh_brand       0.01693 0.0003973
   driv_age_c          0.01 0.0002895
    veh_power      0.007708 0.0002781
      veh_gas      0.006031 0.0007295
       region      0.005207 0.0004607
  log_density      0.002956 0.0003237
```
