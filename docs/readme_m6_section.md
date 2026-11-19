## Risultati della milestone 6 (deploy e monitoraggio)

- Servizio FastAPI (`src/serving/`) con premio puro e triage NLP, immagine Docker a tre fasi, test automatici nell'immagine e job di CI dedicato.
- Scelte di sicurezza dei modelli: Bonus-Malus obbligatorio (mai imputato), categorie sconosciute segnalate, log senza testi, degrado controllato senza stadio 2.
- Monitoraggio sul traffico: PSI degli input, variazione della frequenza media prevista con soglia che tiene conto del rumore, quota di candidati e lunghezza dei testi.
- Dettagli, endpoint e limiti: `docs/serving.md`.
