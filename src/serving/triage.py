"""Triage dei reclami: stadio 1 (TF-IDF + regressione logistica) e, per i candidati, stadio 2 (transformer).

Pulizia del testo: porta in Python le stesse regole di `src/nlp/nhtsa_transforms.clean_narrative` (scritte per Spark);
se si modificano le une vanno modificate le altre. I punteggi sono comparabili con quelli del lake solo con la stessa pulizia.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np

FOIA_RE = re.compile(r"information redacted pursuant to the freedom of information act \(foia\),?\s*5 u\.?s\.?c\.? ?552\(b\)\(6\)\.?",
                     re.I)
UNICODE_SPACES_RE = re.compile("[   -​  　﻿]")
INITIALS_RE = re.compile(r"(\s*\*[A-Za-z]{2,3}\.?)+\s*$")
PREFIX_RE = re.compile(r"^[A-Za-z]{1,4}\*+\s*")
MIN_CHARS, MIN_WORDS = 30, 5  # stessi minimi del silver e del gold
LONG_WORDS = 190  # circa 256 token: oltre, il transformer legge solo l'inizio


def clean_text(text: str) -> str:
    t = UNICODE_SPACES_RE.sub(" ", text)
    t = FOIA_RE.sub("[REDACTED]", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = INITIALS_RE.sub("", t).strip()
    return PREFIX_RE.sub("", t).strip()


def logit(p, eps: float = 1e-7):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


class Stage2(Protocol):
    def score(self, texts: list) -> np.ndarray: ...


class TriageModel:
    def __init__(self, bundle: dict, ref: dict, stage2: Stage2 | None = None):
        self.vec, self.clf = bundle["vectorizer"], bundle["classifier"]
        self.threshold = float(ref["threshold_candidate"])
        self.blend = ref.get("blend")
        self.review_threshold = ref.get("review_threshold")
        self.reference = ref
        self.stage2 = stage2
        self.version = f"tfidf-lr{'+transformer' if stage2 is not None else ''}|{ref.get('created_utc', 'n/d')}"

    @classmethod
    def load(cls, models_dir, ref: dict, stage2: Stage2 | None = None):
        bundle = joblib.load(Path(models_dir) / "nlp_tfidf.joblib")
        return cls(bundle, ref, stage2)

    @staticmethod
    def prepare(text: str) -> tuple:
        """(testo pulito, n_parole). Solleva ValueError se il testo e' troppo corto per il dominio del modello."""
        t = clean_text(text)
        n = len(t.split(" ")) if t else 0
        if len(t) < MIN_CHARS or n < MIN_WORDS:
            raise ValueError(f"testo troppo corto dopo la pulizia ({len(t)} caratteri, {n} parole): "
                             f"minimo {MIN_CHARS} caratteri e {MIN_WORDS} parole")
        return t, n

    def predict(self, texts: list) -> list:
        cleaned, n_words = zip(*[self.prepare(t) for t in texts])
        s1 = self.clf.predict_proba(self.vec.transform(list(cleaned)))[:, 1]
        cand = s1 >= self.threshold
        s2 = np.full(len(s1), np.nan)
        if self.stage2 is not None and cand.any():
            idx = np.where(cand)[0]
            s2[idx] = self.stage2.score([cleaned[i] for i in idx])
        out = []
        for i in range(len(s1)):
            w = []
            if n_words[i] > LONG_WORDS:
                w.append(f"testo lungo ({n_words[i]} parole): il transformer legge solo i primi ~256 token")
            r = {"stage1_score": float(s1[i]), "is_candidate": bool(cand[i]), "stage2_score": None, "score": None,
                 "review": None, "n_words": int(n_words[i]), "warnings": w, "model_version": self.version,
                 "text_hash": hashlib.sha256(cleaned[i].encode("utf-8")).hexdigest()[:16]}
            if not cand[i]:
                r["route"], r["review"] = "not_candidate", False
            elif self.stage2 is None:
                r["route"] = "stage1_only"
                w.append("stadio 2 non disponibile: nessuna decisione di revisione")
            else:
                r["route"] = "cascade"
                r["stage2_score"] = float(s2[i])
                z = s2[i]
                if self.blend:
                    b = self.blend
                    z = 1 / (1 + np.exp(-(b["intercept"] + b["coef_stage1"] * logit(s1[i]) + b["coef_stage2"] * logit(s2[i]))))
                r["score"] = float(z)
                if self.review_threshold is not None:
                    r["review"] = bool(z >= self.review_threshold)
            out.append(r)
        return out


def load_reference(models_dir) -> dict:
    return json.loads((Path(models_dir) / "serving_reference.json").read_text(encoding="utf-8"))
