"""Milestone 4, passo 4: stadio 1 della cascata. Punteggi TF-IDF su tutti i reclami e scelta dei candidati.

Il TF-IDF (gia' addestrato in 03_baseline_tfidf.py) classifica tutti i reclami; i candidati sono il `CANDIDATE_Q`
(default 20%) con il punteggio piu' alto. Solo i candidati vengono riletti dal transformer (stadio 2).
- validation e test: punteggi del modello addestrato sull'intero train; soglia dei candidati = quantile della validation
  (come in produzione: la soglia si fissa prima di vedere i dati nuovi);
- train: punteggi FUORI CAMPIONE (5 fold), altrimenti il TF-IDF avrebbe punteggi troppo ottimistici sui propri dati
  di addestramento e il transformer imparerebbe a correggere errori che in produzione non esistono.

Output: data/nlp/stage1_scores.parquet, data/nlp/stage1_meta.json. Gira nel container spark o nlp.
"""
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

NLP = Path(os.getenv("NLP_DIR", "data/nlp"))
MODELS = Path(os.getenv("MODELS_DIR", "models"))
Q = float(os.getenv("CANDIDATE_Q", "0.20"))

bundle = joblib.load(MODELS / "nlp_tfidf.joblib")
vec, clf = bundle["vectorizer"], bundle["classifier"]
df = pd.read_parquet(NLP / "narratives", columns=["odino", "text", "serious", "split"])
print(f"[dati] {len(df):,} reclami | C del TF-IDF: {clf.C}", flush=True)

t0 = time.time()
X = vec.transform(df["text"])
print(f"[tempo] vettorizzazione: {time.time() - t0:.0f}s", flush=True)
y = df["serious"].to_numpy()
split = df["split"].to_numpy()

score = np.zeros(len(df))
for s in ("validation", "test"):
    m = split == s
    score[m] = clf.predict_proba(X[m])[:, 1]

tr_idx = np.where(split == "train")[0]
t0 = time.time()
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(tr_idx, y[tr_idx])):
    m = LogisticRegression(C=clf.C, solver="liblinear", max_iter=300).fit(X[tr_idx[a]], y[tr_idx[a]])
    score[tr_idx[b]] = m.predict_proba(X[tr_idx[b]])[:, 1]
    print(f"[tempo] fold {k + 1}/5: {time.time() - t0:.0f}s", flush=True)

thr_val = float(np.quantile(score[split == "validation"], 1 - Q))
thr_train = float(np.quantile(score[split == "train"], 1 - Q))  # i punteggi fuori campione hanno scala simile
thr = np.where(split == "train", thr_train, thr_val)
df["score1"] = score
df["is_candidate"] = score >= thr

summ = {}
print(f"\nSoglia candidati (top {Q:.0%} sulla validation): {thr_val:.5f} | train (fuori campione): {thr_train:.5f}")
for s in ("train", "validation", "test"):
    d = df[df["split"] == s]
    c = d[d["is_candidate"]]
    summ[s] = {"reclami": int(len(d)), "candidati": int(len(c)), "quota_candidati": float(len(c) / len(d)),
               "positivi": int(d["serious"].sum()), "positivi_nei_candidati": int(c["serious"].sum()),
               "recall_stadio1": float(c["serious"].sum() / d["serious"].sum()),
               "tasso_grave_nei_candidati": float(c["serious"].mean())}
    print(f"  {s:10s}: candidati {len(c):>7,} su {len(d):>7,} ({len(c) / len(d):.1%}) | "
          f"positivi catturati {c['serious'].sum():>6,} su {d['serious'].sum():>6,} ({c['serious'].sum() / d['serious'].sum():.1%}) "
          f"| tasso grave tra i candidati {c['serious'].mean():.1%}")

df[["odino", "split", "serious", "score1", "is_candidate"]].to_parquet(NLP / "stage1_scores.parquet", index=False)
(NLP / "stage1_meta.json").write_text(json.dumps({"candidate_q": Q, "thr_validation": thr_val, "thr_train_oof": thr_train,
                                                  "summary": summ}, indent=2), encoding="utf-8")
print(f"\n[ok] {NLP / 'stage1_scores.parquet'}")
