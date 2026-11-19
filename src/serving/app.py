"""API FastAPI. Si avvia con `uvicorn serving.app:create_app --factory`.

Variabili: MODELS_DIR (/models), PREDICTION_LOG (/logs/predictions.jsonl), PRICING_SUFFIX (_payment),
STAGE2_DIR (<MODELS_DIR>/nlp_stage2/best), DISABLE_STAGE2=1 per servire solo lo stadio 1.
Se un modello non si carica il servizio parte lo stesso: /health risponde 200 (processo vivo), /ready dice 503 e
l'endpoint relativo risponde 503 con il motivo, cosi' un'orchestrazione sa che cosa non e' disponibile.
"""
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Response

from . import monitor
from .pricing import PricingModel
from .schemas import BatchOut, PolicyBatch, Policy, PremiumOut, TriageIn, TriageOut
from .triage import TriageModel, load_reference

log = logging.getLogger("serving")


def _load_pricing(models_dir, suffix, reference):
    try:
        return PricingModel.load(models_dir, suffix, reference), None
    except Exception as e:  # noqa: BLE001 - si riporta il motivo in /ready
        log.error("premio puro non disponibile: %s", e)
        return None, f"{type(e).__name__}: {e}"


def _load_triage(models_dir, ref, stage2_dir, disable_stage2):
    if ref is None or "nlp" not in ref:
        return None, "riferimento NLP assente (serving_reference.json): eseguire `make serving-export`"
    stage2, note = None, None
    if not disable_stage2 and Path(stage2_dir).exists():
        try:
            from .stage2 import TransformerScorer
            meta_path = Path(models_dir) / "nlp_stage2_metadata.json"
            max_len = json.loads(meta_path.read_text())["max_len"] if meta_path.exists() else 256
            stage2 = TransformerScorer(stage2_dir, max_len=max_len)
        except Exception as e:  # noqa: BLE001
            note = f"stadio 2 non caricato ({type(e).__name__}: {e}): si serve solo lo stadio 1"
            log.error(note)
    try:
        return TriageModel.load(models_dir, ref["nlp"], stage2), note
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def create_app(models_dir=None, log_path=None, stage2=None, pricing_suffix=None) -> FastAPI:
    models_dir = Path(models_dir or os.getenv("MODELS_DIR", "/models"))
    log_path = log_path if log_path is not None else os.getenv("PREDICTION_LOG", "/logs/predictions.jsonl")
    suffix = pricing_suffix if pricing_suffix is not None else os.getenv("PRICING_SUFFIX", "_payment")
    app = FastAPI(title="Insurance Claims Risk Platform", version="0.6.0",
                  description="Premio puro RC Auto (prototipo) e triage dei reclami NHTSA. Non per uso operativo.")

    try:
        reference = load_reference(models_dir)
    except Exception as e:  # noqa: BLE001
        log.warning("serving_reference.json non disponibile: %s", e)
        reference = None
    pricing, pricing_err = _load_pricing(models_dir, suffix, reference.get("pricing") if reference else None)
    if stage2 is not None and reference and "nlp" in reference:  # iniezione per i test
        try:
            triage, triage_note = TriageModel.load(models_dir, reference["nlp"], stage2), None
        except Exception as e:  # noqa: BLE001
            triage, triage_note = None, f"{type(e).__name__}: {e}"
    else:
        triage, triage_note = _load_triage(models_dir, reference, os.getenv("STAGE2_DIR", str(models_dir / "nlp_stage2" / "best")),
                                           os.getenv("DISABLE_STAGE2", "0") == "1")
    plog = monitor.PredictionLog(log_path or None)
    app.state.pricing, app.state.triage, app.state.plog = pricing, triage, plog

    def need_pricing():
        if pricing is None:
            raise HTTPException(503, f"modello di premio non disponibile: {pricing_err}")

    def need_triage():
        if triage is None:
            raise HTTPException(503, f"triage non disponibile: {triage_note}")

    @app.get("/health")
    def health():
        return {"status": "alive"}

    @app.get("/ready")
    def ready(response: Response):
        st = {"pricing": pricing is not None, "triage": triage is not None,
              "triage_stage2": bool(triage and triage.stage2 is not None), "log_write_errors": plog.errors}
        errors = {k: v for k, v in {"pricing": pricing_err, "triage": triage_note}.items() if v}
        if not st["pricing"] or not st["triage"]:
            response.status_code = 503
        return {**st, "notes": errors}

    @app.get("/v1/model-info")
    def model_info():
        return {"pricing": None if pricing is None else {
                    "version": pricing.version, "severity_capped_mean": pricing.severity, "tail_load": pricing.tail_load,
                    "features": {"numeric": pricing.numeric, "categorical": pricing.categorical},
                    "test_metrics": pricing.meta["frequency"].get("test_metrics")},
                "triage": None if triage is None else {
                    "version": triage.version, "candidate_threshold": triage.threshold,
                    "review_threshold": triage.review_threshold,
                    "stage2": triage.stage2 is not None},
                "warning": "Prototipo di ricerca su dati pubblici: non per tariffazione o decisioni reali."}

    def _premium(policies: list, rid: str):
        res, X = pricing.predict(policies)
        outs = []
        for p, r, (_, row) in zip(policies, res, X.iterrows()):
            outs.append(PremiumOut(request_id=rid, **r))
            plog.append("premium", rid, {**row.to_dict(), "exposure": p.exposure}, {
                "expected_frequency": r["expected_frequency"], "pure_premium_per_year": r["pure_premium_per_year"],
                "model_version": r["model_version"]})
        return outs

    @app.post("/v1/premium", response_model=PremiumOut)
    def premium(policy: Policy, response: Response):
        need_pricing()
        rid = uuid.uuid4().hex
        response.headers["X-Request-ID"] = rid
        return _premium([policy], rid)[0]

    @app.post("/v1/premium/batch", response_model=BatchOut)
    def premium_batch(batch: PolicyBatch, response: Response):
        need_pricing()
        rid = uuid.uuid4().hex
        response.headers["X-Request-ID"] = rid
        return BatchOut(request_id=rid, results=_premium(batch.policies, rid))

    @app.post("/v1/triage", response_model=TriageOut)
    def triage_endpoint(body: TriageIn, response: Response):
        need_triage()
        rid = uuid.uuid4().hex
        response.headers["X-Request-ID"] = rid
        try:
            r = triage.predict([body.text])[0]
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        plog.append("triage", rid, {"n_words": r["n_words"], "text_hash": r["text_hash"]}, {
            k: r[k] for k in ("stage1_score", "stage2_score", "score", "is_candidate", "review", "route", "model_version")})
        r.pop("text_hash")
        return TriageOut(request_id=rid, **r)

    @app.get("/monitor/summary")
    def summary(last: int = Query(5000, ge=1, le=100000, description="Ultime N richieste per tipo"),
                last_premium: Optional[int] = Query(None, ge=1, le=100000, description="Finestra solo per il premio"),
                last_triage: Optional[int] = Query(None, ge=1, le=100000, description="Finestra solo per il triage")):
        out = {"log_write_errors": plog.errors, "min_n": monitor.MIN_N}
        if reference and "pricing" in reference:
            out["premium"] = monitor.premium_report(plog.read("premium", last_premium or last), reference["pricing"])
        if reference and "nlp" in reference:
            out["triage"] = monitor.triage_report(plog.read("triage", last_triage or last), reference["nlp"])
        out["note"] = ("Soglie proposte nelle model card (PSI 0,05/0,10; frequenza media +-5%; quota candidati +-5 punti): "
                       "ipotesi da validare. Il PSI sul punteggio non e' usato perche' poco sensibile.")
        return out

    return app
