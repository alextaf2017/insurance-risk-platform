# Servizio di inferenza (milestone 6)

API FastAPI con due funzioni: **premio puro** RC Auto e **triage dei reclami** NHTSA. Prototipo dimostrativo, non per tariffare né decidere.

## Avvio

```bash
make serving-export    # una volta: crea models/serving_reference.json (riferimenti di monitoraggio, soglie, combinazione)
make api-test          # test nell'immagine Docker
make api-up            # build e avvio su http://localhost:8000  (WITH_TORCH=0 make api-up: senza stadio 2)
make api-demo          # traffico reale del test + rapporto di monitoraggio
make api-demo-shift    # stesso, con Bonus-Malus x1,5: il monitoraggio deve scattare
make api-reset-log     # svuota il log delle previsioni
```

Documentazione interattiva su `http://localhost:8000/docs`.

## Endpoint

| Metodo e percorso | Scopo |
|---|---|
| `GET /health` | Il processo è vivo (sempre 200) |
| `GET /ready` | 200 se premio e triage sono caricati, 503 altrimenti, con il motivo |
| `GET /v1/model-info` | Versioni, soglie, metriche di test, avvertenza d'uso |
| `POST /v1/premium` | Una polizza → frequenza, sinistri attesi, severità, fattore di coda, premio puro |
| `POST /v1/premium/batch` | Fino a 1.000 polizze |
| `POST /v1/triage` | Un testo → punteggio stadio 1, candidato sì/no, punteggio stadio 2, decisione di revisione |
| `GET /monitor/summary?last=N` | PSI degli input, variazione della frequenza media prevista, quota di candidati. `last_premium` e `last_triage` fissano finestre diverse |

Esempio:

```bash
curl -s localhost:8000/v1/premium -H 'content-type: application/json' -d '{
  "driv_age": 35, "veh_age": 4, "bonus_malus": 60, "veh_power": 6, "density": 1200,
  "veh_gas": "Regular", "veh_brand": "B12", "region": "R24", "exposure": 1.0}'
```

## Scelte di progetto

- **Premio = frequenza (boosting) × severità costante con cap × fattore di coda**, la stessa composizione valutata nella milestone 3.
  Il servizio rifiuta di avviarsi con un modello di severità diverso dalla costante: non serve un premio che non è stato validato.
- **Bonus-Malus obbligatorio.** Se manca, la richiesta è rifiutata (422) e non imputata: il registro dei rischi (#3) mostra che con il
  Bonus-Malus sbagliato la devianza peggiora del 7,1% e il portafoglio è sottostimato di circa il 30%. Un modello di riserva senza
  Bonus-Malus non esiste ancora.
- **Stesse regole del lake.** Limiti 20 (età veicolo), 90 (età conducente), 150 (Bonus-Malus) e logaritmo della densità, con avviso nella risposta
  quando scattano. Le categorie si abbinano ai livelli del modello senza distinguere maiuscole e apici (nel silver `veh_gas` ha apici letterali);
  una categoria sconosciuta genera un avviso e un valore mancante, non un errore.
- **Pulizia dei testi** portata in Python (`serving/triage.py`) dalle regole Spark di `nhtsa_transforms.py`. Sono due copie da tenere allineate:
  un test verifica i casi principali (sigle in coda, prefisso `TL*`, spazi unicode, oscuramento FOIA), non l'equivalenza completa.
- **Cascata**: lo stadio 2 gira solo sui candidati dello stadio 1. La decisione `review` usa la combinazione logistica dei due punteggi
  (addestrata sulla validation) e una soglia che fa rivedere il 5% dei reclami, scelta sulla validation e verificata sul test da `make serving-export`.
  Senza PyTorch o checkpoint il servizio degrada allo stadio 1 e non emette decisioni di revisione (`route: stage1_only`).
- **Privacy**: il log delle previsioni non contiene il testo dei reclami, solo lunghezza, hash troncato (16 caratteri) e punteggi.
- **Il log non blocca le previsioni**: se la scrittura fallisce si conta l'errore (visibile in `/ready`) e si risponde comunque.
- **Immagine**: tre fasi (base, test, runtime), utente non root, `HF_HUB_OFFLINE=1`, versione di scikit-learn uguale a quella dei metadati di addestramento
  (i modelli sono pickle). `.dockerignore` esclude `data/` e `models/` dal contesto di build; i modelli si montano in sola lettura.

## Monitoraggio

`/monitor/summary` applica le soglie proposte nelle model card (ipotesi da validare):

- PSI di ogni variabile di input contro il train: avviso ≥ 0,05, azione ≥ 0,10.
- Frequenza media prevista: azione se la variazione supera ±5% **e** tre errori standard, calcolati dalla sola numerosità del campione;
  con la regola del 5% da sola, a 500 richieste il rumore produrrebbe falsi allarmi frequenti.
- Triage: quota di candidati (±5 punti da 20%), lunghezza media (> 140 parole) e PSI sulle parole (> 0,10): solo avvisi, perché senza etichette non si misura la recall.
- Il log è cumulativo: il rapporto copre le ultime N richieste, quindi mescola esecuzioni diverse se non si delimita la finestra (il demo lo fa da solo).
- Sotto 500 richieste lo stato è `insufficient_data`: il PSI a 10 fasce con pochi dati è dominato dal rumore.

## Limiti noti

- Lo stadio 2 (`serving/stage2.py`) non è coperto da test automatici (servono PyTorch e un checkpoint): si verifica con `make api-demo`.
  I test usano modelli minuscoli su dati sintetici e uno stadio 2 finto: verificano il contratto e la logica, non la qualità dei modelli.
- Log in file JSONL e rapporto calcolato rileggendo il file: va bene per migliaia di richieste, non per un servizio reale (servirebbero una tabella o uno stream).
- Nessuna autenticazione, limiti di frequenza o versionamento dei modelli a caldo.
- I modelli sono quelli della milestone 3-4: ereditano tutti i limiti delle model card (target dei sinistri con importo, coda, assenza di date).
- Le latenze reali dipendono dalla macchina: le misura `make api-demo`.
