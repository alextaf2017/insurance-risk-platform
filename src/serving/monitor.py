"""Log delle previsioni e rapporto di monitoraggio.

Privacy: per il triage NON si salva il testo (puo' contenere dati personali): solo lunghezza, hash troncato e punteggi.
Il log e' un file JSONL: sufficiente per la dimostrazione; in produzione andrebbe in una tabella o in un sistema di eventi.
Un errore di scrittura del log non deve mai far fallire una previsione.
"""
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

from . import reference as ref

log = logging.getLogger("serving.monitor")
MIN_N = 500  # sotto questa soglia il PSI e' dominato dal rumore (distorsione attesa ~ fasce / n)
ORDER = {"insufficient_data": 0, "ok": 1, "warn": 2, "action": 3}


class PredictionLog:
    def __init__(self, path):
        self.path = Path(path) if path else None
        self.errors = 0
        self._lock = threading.Lock()
        if self.path:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                log.warning("cartella del log non scrivibile: %s", e)

    def append(self, kind: str, request_id: str, inputs: dict, outputs: dict):
        if not self.path:
            return
        rec = {"ts": round(time.time(), 3), "kind": kind, "request_id": request_id, "inputs": inputs, "outputs": outputs}
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            self.errors += 1
            log.warning("scrittura del log fallita: %s", e)

    def read(self, kind: str, last: int = 5000) -> list:
        if not self.path or not self.path.exists():
            return []
        out = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == kind:
                    out.append(r)
        return out[-last:]


def _worst(statuses):
    return max(statuses, key=lambda s: ORDER[s]) if statuses else "insufficient_data"


def premium_report(records: list, reference: dict, rate_tol: float = 0.05) -> dict:
    """Confronta gli ultimi input e le frequenze previste con il riferimento del train."""
    n = len(records)
    rep = {"n": n, "status": "insufficient_data", "features": {}, "mean_rate": None, "mean_rate_change": None}
    if n == 0:
        return rep
    rates = np.array([r["outputs"]["expected_frequency"] for r in records])
    expo = np.array([r["inputs"].get("exposure", 1.0) for r in records])
    rep["mean_rate"] = float(np.average(rates, weights=expo))
    if n < MIN_N:
        rep["note"] = f"meno di {MIN_N} richieste: PSI non affidabile"
        return rep
    sts = []
    ref_rate = reference.get("mean_rate")
    if ref_rate:
        rep["mean_rate_change"] = rep["mean_rate"] / ref_rate - 1
        # errore standard del rapporto dovuto alla sola numerosita' del campione: con poche richieste una variazione
        # del 5% puo' essere rumore, quindi serve anche che superi 3 errori standard (circa lo 0,3% di falsi allarmi per controllo)
        se = float(np.std(rates) / rep["mean_rate"] / np.sqrt(n))
        rep["mean_rate_change_se"] = round(se, 4)
        sts.append("action" if abs(rep["mean_rate_change"]) >= rate_tol and abs(rep["mean_rate_change"]) > 3 * se else "ok")
    for name, spec in reference["numeric"].items():
        v = [r["inputs"][name] for r in records]
        psi = ref.psi_numeric(spec, v)
        rep["features"][name] = {"psi": round(psi, 4), "status": ref.status_from_psi(psi)}
    for name, spec in reference["categorical"].items():
        psi = ref.psi_categorical(spec, [r["inputs"][name] for r in records])
        rep["features"][name] = {"psi": round(psi, 4), "status": ref.status_from_psi(psi)}
    sts += [f["status"] for f in rep["features"].values()]
    rep["status"] = _worst(sts)
    return rep


def triage_report(records: list, reference: dict, share_tol: float = 0.05, max_words: float = 140.0) -> dict:
    """Quota di candidati e lunghezza dei testi rispetto al riferimento. Solo avvisi: senza etichette non si misura la recall."""
    n = len(records)
    rep = {"n": n, "status": "insufficient_data", "candidate_share": None, "mean_words": None, "psi_words": None}
    if n == 0:
        return rep
    words = [r["inputs"]["n_words"] for r in records]
    share = float(np.mean([r["outputs"]["is_candidate"] for r in records]))
    rep.update(candidate_share=share, mean_words=float(np.mean(words)))
    if n < MIN_N:
        rep["note"] = f"meno di {MIN_N} richieste: indicatori non affidabili"
        return rep
    sts = ["warn" if abs(share - reference["candidate_share"]) > share_tol else "ok",
           "warn" if np.mean(words) > max_words else "ok"]
    rep["psi_words"] = round(ref.psi_numeric(reference["words"], words), 4)
    sts.append("warn" if rep["psi_words"] > 0.10 else "ok")
    rep["status"] = _worst(sts)
    return rep
