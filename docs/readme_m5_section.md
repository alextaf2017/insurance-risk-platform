## Risultati della milestone 5 (rischio e IA responsabile)

- **Calibrazione per gruppo**: su 35 gruppi il boosting ne ha 1 con intervallo che esclude la calibrazione (35-49 anni, +4,7% sottostima), il GLM 3 (sottostime 4-14%). Non dimostra equità.
- **Guasto di una variabile**: Bonus-Malus errato costa +7,1% di devianza e dimezza il Gini (0,318 → 0,155); età del conducente +1,3%.
- **Overfitting**: scarto train-test 3,6%; il GLM è migliore sotto circa 27.000 polizze di addestramento, il boosting sopra.
- **Monitoraggio**: il PSI del punteggio è poco sensibile (+6% di premio medio → 0,011); si usano PSI di variabile e variazione della frequenza media prevista.
- **Drift NLP 2018-2026**: i testi si allungano (91 → 122 parole), la recall al 5% resta 0,82-0,86; soglie di allarme riviste.
- Documenti: `docs/model_card_insurance_frequency.md`, `docs/model_card_nlp_triage.md`, `docs/risk_register.md`; report in `reports/m5_*`.
