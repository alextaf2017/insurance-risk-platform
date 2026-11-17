# Milestone 4: controllo del secondo stadio (transformer contro lessicale specializzato)

Esempi di addestramento del transformer e della variante (a): 9,000 (3,000 positivi). Campione pesato del test: 10,758 righe. Bootstrap B=200.

## 1. PR-AUC tra i candidati e recall al tasso di revisione (sull'intero test)

```text
                                                ranker  pr_auc_candidati  ic_low  ic_high  recall_2%  recall_5%  recall_10%
                                Solo TF-IDF (stadio 1)            0.7554   0.737   0.7723      0.593     0.8353      0.9137
                                      Solo transformer            0.7849  0.7671   0.8015     0.6139      0.848      0.9213
                     Lessicale stadio 2 (9,000 esempi)            0.7134  0.6939   0.7301      0.572     0.8072      0.8941
         Lessicale stadio 2 (tutti i 71,722 candidati)            0.7606  0.7427   0.7761     0.5955     0.8398      0.9157
                                  TF-IDF + transformer            0.7964  0.7805   0.8129      0.621     0.8681      0.9281
TF-IDF + Lessicale stadio 2 (tutti i 71,722 candidati)            0.7607  0.7429   0.7764     0.5969     0.8392      0.9157
```

## 2. Confronti decisivi (differenza A - B con IC bootstrap al 95%; l'effetto e' reale se l'IC esclude lo zero)
```text
                                                     A                                                      B          metrica     diff     ic_low  ic_high
                                      Solo transformer          Lessicale stadio 2 (tutti i 71,722 candidati) PR-AUC candidati   0.0243    0.01169  0.03715
                                      Solo transformer          Lessicale stadio 2 (tutti i 71,722 candidati)     recall al 2%   0.0184   0.007904  0.03085
                                      Solo transformer          Lessicale stadio 2 (tutti i 71,722 candidati)     recall al 5% 0.008208 -0.0008562  0.01728
                                      Solo transformer          Lessicale stadio 2 (tutti i 71,722 candidati)    recall al 10% 0.005661  -0.002554  0.01359
                                      Solo transformer                      Lessicale stadio 2 (9,000 esempi) PR-AUC candidati  0.07145    0.05658  0.08567
                                      Solo transformer                      Lessicale stadio 2 (9,000 esempi)     recall al 2%  0.04189    0.03167  0.05637
                                      Solo transformer                      Lessicale stadio 2 (9,000 esempi)     recall al 5%  0.04076    0.03084  0.05209
                                      Solo transformer                      Lessicale stadio 2 (9,000 esempi)    recall al 10%  0.02717    0.01697  0.03453
                                  TF-IDF + transformer TF-IDF + Lessicale stadio 2 (tutti i 71,722 candidati) PR-AUC candidati  0.03575    0.02956  0.04335
                                  TF-IDF + transformer TF-IDF + Lessicale stadio 2 (tutti i 71,722 candidati)     recall al 2%  0.02406    0.01444  0.03455
                                  TF-IDF + transformer TF-IDF + Lessicale stadio 2 (tutti i 71,722 candidati)     recall al 5%  0.02887    0.02009  0.03709
                                  TF-IDF + transformer TF-IDF + Lessicale stadio 2 (tutti i 71,722 candidati)    recall al 10%  0.01245   0.006227  0.01896
TF-IDF + Lessicale stadio 2 (tutti i 71,722 candidati)                                 Solo TF-IDF (stadio 1) PR-AUC candidati 0.005308   0.002561 0.008196
TF-IDF + Lessicale stadio 2 (tutti i 71,722 candidati)                                 Solo TF-IDF (stadio 1)     recall al 2% 0.003963  -0.005944 0.009631
TF-IDF + Lessicale stadio 2 (tutti i 71,722 candidati)                                 Solo TF-IDF (stadio 1)     recall al 5% 0.003963  -0.001132  0.01076
TF-IDF + Lessicale stadio 2 (tutti i 71,722 candidati)                                 Solo TF-IDF (stadio 1)    recall al 10% 0.001981  -0.002279 0.006241
```

Lettura: se 'Solo transformer' non batte il lessicale di stadio 2 (IC che include lo zero), il vantaggio e' dovuto alla specializzazione sulla regione dei candidati e non al transformer; se lo batte, il modello contestuale aggiunge informazione che il lessico non ha.
