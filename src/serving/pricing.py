"""Premio puro = frequenza (gradient boosting, sinistri pagati) x severita' media con cap x fattore di coda.

E' la stessa composizione valutata nella milestone 3 ("freq HGB x sev"). La severita' e' una costante: in
cross-validation nessun modello ha battuto la costante di almeno l'1%. Supporta solo questa scelta e fallisce se i
metadati ne dicono un'altra, per non servire un premio diverso da quello validato.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .features import build_frame


class PricingModel:
    def __init__(self, pipe, freq_meta: dict, price_meta: dict, reference: dict | None):
        if price_meta.get("severity_model") != "constant":
            raise ValueError(f"severity_model={price_meta.get('severity_model')!r}: il servizio supporta solo 'constant'")
        self.pipe = pipe
        self.numeric = freq_meta["features"]["numeric"]
        self.categorical = freq_meta["features"]["categorical"]
        enc = pipe.named_steps["prep"].named_transformers_["cat"]
        self.levels = {c: [str(x) for x in cats] for c, cats in zip(self.categorical, enc.categories_)}
        self.severity = float(price_meta["severity_constant_capped"])
        self.tail_load = float(price_meta["tail_load"])
        self.reference = reference
        self.version = f"hgb-{freq_meta.get('created_utc', 'n/d')}|sklearn-{freq_meta.get('sklearn_version', 'n/d')}"
        self.meta = {"frequency": freq_meta, "pricing": price_meta}

    @classmethod
    def load(cls, models_dir, suffix: str = "_payment", reference: dict | None = None):
        d = Path(models_dir)
        pipe = joblib.load(d / f"frequency_hgb{suffix}.joblib")
        fm = json.loads((d / f"frequency_metadata{suffix}.json").read_text(encoding="utf-8"))
        pm = json.loads((d / f"pricing_metadata{suffix}.json").read_text(encoding="utf-8"))
        return cls(pipe, fm, pm, reference)

    def predict(self, policies: list) -> tuple:
        """(righe di risultato, DataFrame delle feature usate). Le richieste sono gia' validate dallo schema."""
        X, warns = build_frame(policies, self.numeric, self.categorical, self.levels)
        rate = np.asarray(self.pipe.predict(X), dtype=float)
        out = []
        for p, r, w in zip(policies, rate, warns):
            expo = float(p.exposure if hasattr(p, "exposure") else p.get("exposure", 1.0))
            per_year = float(r * self.severity * self.tail_load)
            out.append({"expected_frequency": float(r), "expected_claims": float(r * expo),
                        "severity_capped_mean": self.severity, "tail_load": self.tail_load,
                        "pure_premium_per_year": per_year, "pure_premium": per_year * expo, "warnings": w,
                        "model_version": self.version})
        return out, X
