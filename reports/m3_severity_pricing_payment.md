# Milestone 3B: severita', premio puro e benchmark Tweedie (frequenza: sev_claim_count)

Sinistri train: 21,113 | test: 5,331 | cap al percentile 0.995 del train = 33,582 | tail load = 1.385
Importo medio per sinistro (senza cap): train 2,329 | test 2,013  <- la differenza mostra quanto la coda destabilizzi le medie

Diagnosi di coerenza tra frequenza e severita':
- sinistri contati (claim_nb): train 28,826 | test 7,230
- sinistri con importo (tabella severita'): train 21,113 | test 5,331 (rapporto train 0.732 | test 0.737)
Un rapporto sotto 1 vuol dire che una parte dei sinistri contati non ha un importo registrato: frequenza (claim_nb) x severita' (solo sinistri con importo) sovrastimerebbe il costo di circa 1/rapporto.

## 1. Scelta del modello di severita' in cross-validation (5 gruppi sul train, importi con cap)

```text
            model  cv_gamma_dev  vs_costante_%
         Costante         1.167              0
Gamma GLM (tuned)         1.164         0.2572
HGB gamma (tuned)         1.162         0.4622
```

Regola: si usa un modello solo se migliora la devianza CV di almeno l'1%. Migliore: HGB con +0.46%. **Scelta: constant**.

## 2. Sensibilita' al cap: guadagno CV del Gamma GLM sulla costante, per soglia

```text
 cap_quantile  cap_value  tail_load  GLM_gain_vs_const_%
         0.99      16254      1.489               0.3781
        0.995      33582      1.385               0.2512
        0.999     152305       1.21             -0.07872
```
Se il guadagno cambia molto con la soglia, la segmentazione della severita' dipende dalle poche perdite estreme e non e' affidabile.

## 3. Severita' sul test (sinistri con cap)

Devianza Gamma media: costante 1.2228 | modello scelto (constant) 1.2228 | guadagno +0.0000 (IC bootstrap 95%: +0.0000, +0.0000).

## 4. Premio puro sul test: perdita reale (senza cap) contro prevista

`actual/pred` vicino a 1 = portafoglio ben calibrato; con code pesanti l'IC bootstrap e' troppo stretto, quindi si guarda anche `capped_act/pred` (perdita con cap per sinistro), molto meno rumoroso. `gini_loss` = capacita' di ordinare le polizze per costo atteso.
```text
                model  actual_total  predicted_total  actual/pred  ci_low  ci_high  capped_act/pred  cap_ci_low  cap_ci_high  gini_loss
       freq GLM x sev     1.073e+07        1.223e+07       0.8769  0.7853   0.9635            1.032      0.9742        1.092     0.3136
       freq HGB x sev     1.073e+07        1.228e+07       0.8736  0.7759   0.9552            1.028      0.9612        1.082     0.3304
Tweedie (1.9) diretto     1.073e+07        1.247e+07       0.8601  0.7704   0.9623            1.014      0.9551        1.068     0.3269
```

## 5. Calibrazione per decili del premio puro (freq HGB x sev): perdita osservata vs prevista per anno-polizza

```text
 decile  exposure  observed  predicted  obs/pred
      1      7130     114.5      67.31     1.701
      2      7130     76.06      87.34    0.8708
      3      7131     76.67      100.2    0.7653
      4      7131     69.66      113.1    0.6158
      5      7130     116.1      126.5    0.9182
      6      7130     95.97      141.1      0.68
      7      7131     120.5      160.5    0.7508
      8      7130     167.7      189.8    0.8836
      9      7130       164      245.4    0.6683
     10      7130     503.4      491.2     1.025
```
