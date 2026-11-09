# Milestone 3B: severita', premio puro e benchmark Tweedie (frequenza: claim_nb)

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
       freq GLM x sev     1.073e+07        1.672e+07       0.6416  0.5742    0.707           0.7554      0.7145       0.8005     0.2889
       freq HGB x sev     1.073e+07        1.686e+07       0.6364  0.5646   0.6968           0.7493      0.7012       0.7903     0.3235
Tweedie (1.9) diretto     1.073e+07        1.248e+07       0.8594  0.7697   0.9616            1.013      0.9544        1.067     0.3281
```

## 5. Calibrazione per decili del premio puro (Tweedie (1.9) diretto): perdita osservata vs prevista per anno-polizza

```text
 decile  exposure  observed  predicted  obs/pred
      1      7130     45.86      57.05    0.8039
      2      7130       134      80.83     1.657
      3      7131     67.18      96.24     0.698
      4      7130     92.74      110.1    0.8422
      5      7130     107.6      124.3    0.8654
      6      7130     100.4      140.8    0.7133
      7      7131     121.9      162.4    0.7503
      8      7130     187.1      195.6    0.9563
      9      7131       191      258.6    0.7385
     10      7130       457      524.9    0.8706
```
