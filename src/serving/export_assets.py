"""Costruisce `models/serving_reference.json` (riferimenti di monitoraggio e parametri della cascata NLP).

Gira nel container spark (pandas, scikit-learn, pyarrow). Legge i dati del train e i punteggi gia' calcolati;
il file e' piccolo e senza dati grezzi, quindi si puo' versionare come gli altri metadati.

Premio: distribuzioni delle feature sul train e frequenza media prevista (pesata per esposizione).
NLP: soglia dei candidati (stage1_meta.json), combinazione logistica dei due punteggi addestrata SOLO sulla validation
(come in 07_compare_cascade.py), soglia di revisione al 5% dei reclami calcolata sulla validation e verifica sul test.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from serving import reference as ref
from serving.features import norm  # noqa: F401  (stessa normalizzazione usata a runtime)
from serving.triage import logit

MODELS = Path(os.getenv("MODELS_DIR", "models"))
MODELING = Path(os.getenv("MODELING_DIR", "data/modeling"))
NLP = Path(os.getenv("NLP_DIR", "data/nlp"))
SUFFIX = os.getenv("PRICING_SUFFIX", "_payment")
REVIEW_RATE = float(os.getenv("REVIEW_RATE", "0.05"))


def pricing_section(max_rows: int = 200_000) -> dict:
    fm = json.loads((MODELS / f"frequency_metadata{SUFFIX}.json").read_text(encoding="utf-8"))
    num, cat = fm["features"]["numeric"], fm["features"]["categorical"]
    pol = pd.read_parquet(MODELING / "policies")
    pol = pol[pol["split"] == "train"].reset_index(drop=True)
    for c in cat:
        pol[c] = pol[c].astype(str)
    for c in num + ["exposure"]:
        pol[c] = pol[c].astype("float64")
    sample = pol.sample(min(max_rows, len(pol)), random_state=0)
    pipe = joblib.load(MODELS / f"frequency_hgb{SUFFIX}.joblib")
    rate = pipe.predict(sample[cat + num])
    return {"numeric": {c: ref.numeric_reference(pol[c]) for c in num},
            "categorical": {c: ref.categorical_reference(pol[c]) for c in cat},
            "mean_rate": float(np.average(rate, weights=sample["exposure"])), "n_train": int(len(pol))}


def nlp_section() -> dict:
    meta1 = json.loads((NLP / "stage1_meta.json").read_text(encoding="utf-8"))
    nar = pd.read_parquet(NLP / "narratives", columns=["odino", "text", "serious", "split"])
    pred = pd.read_parquet(NLP / "pred_stage2.parquet")
    val, te = pred[pred["split"] == "validation"], pred[pred["split"] == "test"].copy()
    xv = np.column_stack([logit(val["score1"]), logit(val["score2"])])
    blend = LogisticRegression(C=1.0).fit(xv, val["serious"], sample_weight=val["w"])
    b = {"intercept": float(blend.intercept_[0]), "coef_stage1": float(blend.coef_[0][0]), "coef_stage2": float(blend.coef_[0][1])}

    def score(d):
        return blend.predict_proba(np.column_stack([logit(d["score1"]), logit(d["score2"])]))[:, 1]

    n_val = int((nar["split"] == "validation").sum())
    sv = score(val)
    order = np.argsort(-sv)
    cum = np.cumsum(val["w"].to_numpy()[order])
    k = int(np.searchsorted(cum, REVIEW_RATE * n_val))
    thr = float(sv[order][min(k, len(order) - 1)])
    # verifica sul test (non usato per scegliere nulla)
    n_te, p_te = int((nar["split"] == "test").sum()), int(nar[nar["split"] == "test"]["serious"].sum())
    st = score(te)
    rev = st >= thr
    check = {"reviewed_share": float(te["w"][rev].sum() / n_te),
             "recall": float((te["w"] * te["serious"])[rev].sum() / p_te)}
    words = nar[nar["split"] == "train"]["text"].str.split(" ").str.len().to_numpy()
    return {"threshold_candidate": float(meta1["thr_validation"]), "candidate_q": float(meta1["candidate_q"]),
            "candidate_share": float(meta1["candidate_q"]), "blend": b, "review_rate": REVIEW_RATE,
            "review_threshold": thr, "check_test": check, "words": ref.numeric_reference(words),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def main():
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out["pricing"] = pricing_section()
    print(f"[premio] frequenza media prevista (train): {out['pricing']['mean_rate']:.4f}")
    if (NLP / "pred_stage2.parquet").exists() and (NLP / "stage1_meta.json").exists():
        out["nlp"] = nlp_section()
        c = out["nlp"]["check_test"]
        print(f"[nlp] soglia revisione {out['nlp']['review_threshold']:.4f} | test: rivisti {c['reviewed_share']:.2%} dei reclami, "
              f"recall {c['recall']:.3f}")
    else:
        print("[nlp] pred_stage2.parquet o stage1_meta.json assenti: sezione NLP non scritta")
    (MODELS / "serving_reference.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[ok] {MODELS / 'serving_reference.json'}")


if __name__ == "__main__":
    main()
