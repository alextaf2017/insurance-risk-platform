"""Milestone 4, passo 7: controllo. Il vantaggio viene dal transformer o dal secondo stadio specializzato?

Il transformer e' addestrato solo sui candidati dello stadio 1 (esempi difficili). Per separare i due effetti si
addestra, sugli STESSI esempi, un secondo stadio lessicale (TF-IDF + regressione logistica), e lo si confronta con il
transformer alle stesse condizioni (stesso campione pesato del test, stesse metriche, bootstrap appaiato).
Varianti lessicali: (a) gli stessi 9.000 esempi del transformer; (b) tutti i candidati del train (~72.000).

Richiede 05_stage1_scores.py e 06_finetune_stage2.py. Output: reports/m4_control_stage2.md.
Gira nel container spark o nlp (pandas e scikit-learn).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from ft_utils import logit, recall_at, stratified_sample, weighted_ap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

NLP = Path(os.getenv("NLP_DIR", "data/nlp"))
REPORTS = Path(os.getenv("REPORTS_DIR", "reports"))
N_POS, N_NEG, SEED = int(os.getenv("N_POS", "3000")), int(os.getenv("N_NEG", "6000")), int(os.getenv("SEED", "42"))
B = int(os.getenv("BOOTSTRAP_B", "200"))
PRED = os.getenv("PRED_FILE", "pred_stage2.parquet")
lines = []


def log(text=""):
    print(text, flush=True)
    lines.append(text)


def block(df):
    log("```text")
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}", "display.width", 200):
        log(df.to_string(index=False))
    log("```")


nar = pd.read_parquet(NLP / "narratives", columns=["odino", "text", "serious", "split"])
s1 = pd.read_parquet(NLP / "stage1_scores.parquet", columns=["odino", "score1", "is_candidate"])
pred = pd.read_parquet(NLP / PRED)
df = nar.merge(s1, on="odino", how="inner")
cand_train = df[(df["split"] == "train") & df["is_candidate"]].reset_index(drop=True)
sample = stratified_sample(cand_train, "serious", N_POS, N_NEG, SEED)  # lo stesso campione di 06_finetune_stage2.py

texts = nar[["odino", "text"]]
val = pred[pred["split"] == "validation"].merge(texts, on="odino").reset_index(drop=True)
te = pred[pred["split"] == "test"].merge(texts, on="odino").reset_index(drop=True)
y, w = te["serious"].to_numpy(), te["w"].to_numpy()
all_te = nar[nar["split"] == "test"]
N, P = len(all_te), int(all_te["serious"].sum())

lex = {}
for name, train in [(f"Lessicale stadio 2 ({len(sample):,} esempi)", sample),
                    (f"Lessicale stadio 2 (tutti i {len(cand_train):,} candidati)", cand_train)]:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(train["text"])
    Xv, Xt = vec.transform(val["text"]), vec.transform(te["text"])
    best = (-1.0, None, None)
    for C in (1.0, 3.0, 10.0):
        clf = LogisticRegression(C=C, solver="liblinear", max_iter=300).fit(X, train["serious"])
        ap = weighted_ap(val["serious"], clf.predict_proba(Xv)[:, 1], val["w"])
        if ap > best[0]:
            best = (ap, C, clf)
    _, C, clf = best
    lex[name] = (clf.predict_proba(Xv)[:, 1], clf.predict_proba(Xt)[:, 1], C)
    print(f"[ok] {name}: C={C}", flush=True)


def blend(cols_val, cols_te):
    Xv = np.column_stack([logit(c) for c in cols_val])
    Xt = np.column_stack([logit(c) for c in cols_te])
    m = LogisticRegression(C=1.0).fit(Xv, val["serious"], sample_weight=val["w"])
    return m.predict_proba(Xt)[:, 1]


(n_a, (v_a, t_a, _)), (n_b, (v_b, t_b, _)) = list(lex.items())
scores = {
    "Solo TF-IDF (stadio 1)": te["score1"].to_numpy(),
    "Solo transformer": te["score2"].to_numpy(),
    n_a: t_a,
    n_b: t_b,
    "TF-IDF + transformer": blend([val["score1"], val["score2"]], [te["score1"], te["score2"]]),
    f"TF-IDF + {n_b}": blend([val["score1"], v_b], [te["score1"], t_b]),
}

log("# Milestone 4: controllo del secondo stadio (transformer contro lessicale specializzato)")
log()
log(f"Esempi di addestramento del transformer e della variante (a): {len(sample):,} ({int(sample['serious'].sum()):,} positivi). "
    f"Campione pesato del test: {len(te):,} righe. Bootstrap B={B}.")

rng = np.random.default_rng(42)
idx = [rng.integers(0, len(te), len(te)) for _ in range(B)]


def ci(vals):
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


rows = []
for name, s in scores.items():
    bs = np.array([weighted_ap(y[i], s[i], w[i]) for i in idx])
    lo, hi = ci(bs)
    r = {"ranker": name, "pr_auc_candidati": weighted_ap(y, s, w), "ic_low": lo, "ic_high": hi}
    for rate in (0.02, 0.05, 0.10):
        r[f"recall_{int(rate * 100)}%"] = recall_at(y, s, w, N, P, rate)
    rows.append(r)
log()
log("## 1. PR-AUC tra i candidati e recall al tasso di revisione (sull'intero test)")
log()
block(pd.DataFrame(rows))

log()
log("## 2. Confronti decisivi (differenza A - B con IC bootstrap al 95%; l'effetto e' reale se l'IC esclude lo zero)")
pairs = [("Solo transformer", n_b), ("Solo transformer", n_a), ("TF-IDF + transformer", f"TF-IDF + {n_b}"),
         (f"TF-IDF + {n_b}", "Solo TF-IDF (stadio 1)")]
drows = []
for a, b in pairs:
    sa, sb = scores[a], scores[b]
    d_ap = np.array([weighted_ap(y[i], sa[i], w[i]) - weighted_ap(y[i], sb[i], w[i]) for i in idx])
    lo, hi = ci(d_ap)
    drows.append({"A": a, "B": b, "metrica": "PR-AUC candidati", "diff": weighted_ap(y, sa, w) - weighted_ap(y, sb, w),
                  "ic_low": lo, "ic_high": hi})
    for rate in (0.02, 0.05, 0.10):
        d = np.array([recall_at(y[i], sa[i], w[i], N, P, rate) - recall_at(y[i], sb[i], w[i], N, P, rate) for i in idx])
        lo, hi = ci(d)
        drows.append({"A": a, "B": b, "metrica": f"recall al {int(rate * 100)}%",
                      "diff": recall_at(y, sa, w, N, P, rate) - recall_at(y, sb, w, N, P, rate), "ic_low": lo, "ic_high": hi})
block(pd.DataFrame(drows))
log()
log("Lettura: se 'Solo transformer' non batte il lessicale di stadio 2 (IC che include lo zero), il vantaggio e' dovuto "
    "alla specializzazione sulla regione dei candidati e non al transformer; se lo batte, il modello contestuale aggiunge "
    "informazione che il lessico non ha.")

(REPORTS / "m4_control_stage2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nReport scritto in {REPORTS / 'm4_control_stage2.md'}")
