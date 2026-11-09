# Milestone 3A: modelli di frequenza (target: sev_claim_count)

Train: 543,012 polizze | test: 135,001 polizze | trial Optuna: 25 | bootstrap B=200 | frequenza di riferimento (train): 0.0735

## Risultati sul test (devianza di Poisson per anno-polizza: piu' bassa = meglio)

`gap_%` = degrado da train a test (indizio di overfitting); `D2_test_%` = quota di devianza spiegata rispetto al modello costante; `gini_test` = potere discriminante (0 = nessuno).
```text
                             model  dev_train  dev_test  gap_%  D2_test_%  gini_test
                              Null     0.4782    0.4844   1.29          0          0
                     GLM (default)     0.4559    0.4634  1.637      4.338     0.2794
                       GLM (tuned)     0.4537    0.4616  1.738      4.713     0.2897
GLM (area al posto di log_density)     0.4537    0.4616  1.753      4.699     0.2896
                     HGB (default)     0.4576    0.4761  4.055      1.708     0.3107
                       HGB (tuned)     0.4352    0.4535  4.193      6.382     0.3175
```

## Parametri scelti da Optuna (CV a 3 gruppi su un sottocampione del train)

- GLM: {"n_knots": 7, "alpha": 1e-06} (CV: 0.45525)
- HGB: {"learning_rate": 0.0938, "max_leaf_nodes": 37, "min_samples_leaf": 53, "l2_regularization": 7.5795} (CV: 0.44936)

## GLM (tuned) contro HGB (tuned): differenza di devianza, IC bootstrap al 95% sulle polizze del test

Delta = devianza GLM - devianza HGB = +0.00808 per anno-polizza (IC: +0.00584, +0.01096). Negativo = GLM meglio; positivo = HGB meglio; se l'IC contiene 0 la differenza non e' distinguibile dal rumore.

## Calibrazione per decili di esposizione sul test (HGB (tuned): tasso di sinistri osservato vs previsto)

```text
 decile  exposure  observed  predicted  obs/pred
      1      7130   0.03001    0.02889     1.039
      2      7130   0.03927    0.03749     1.047
      3      7131   0.04544    0.04301     1.056
      4      7131   0.05315    0.04856     1.095
      5      7130   0.05469    0.05429     1.007
      6      7130   0.06241    0.06058      1.03
      7      7131    0.0683     0.0689    0.9913
      8      7130   0.08064    0.08148    0.9898
      9      7130    0.1025     0.1054     0.973
     10      7130    0.2112     0.2109     1.002
```

## Importanza per permutazione (HGB tuned, 60.000 polizze del test): aumento di devianza se si mescola la variabile

```text
      feature  dev_increase       std
bonus_malus_c       0.04676  0.001835
   driv_age_c      0.006153 0.0004035
    veh_brand      0.003168 0.0002074
       region      0.003022 0.0005115
  log_density      0.002566 8.164e-05
    veh_age_c      0.002491 0.0006385
      veh_gas      0.001162 6.641e-05
    veh_power     0.0007364  0.000111
```
