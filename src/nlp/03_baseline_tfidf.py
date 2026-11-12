"""Milestone 4, passo 2: baseline lessicale per il triage delle narrative (esito grave).

Confronta due riferimenti: una regola a parole chiave e un TF-IDF (unigrammi e bigrammi) con regressione
logistica addestrato sull'intero train. Valuta su validation (scelta di C) e test (risultato finale, una volta).
Include i controlli di leakage: parole con il coefficiente piu' alto, tasso di esito grave per lunghezza del
testo e per prime e ultime parole della narrativa.

Output: reports/m4_baseline_tfidf.md, reports/figures/m4_pr_curve_baseline.png, models/nlp_tfidf.joblib,
models/nlp_tfidf_metadata.json, data/nlp/pred_tfidf.parquet (punteggi su validation e test per il confronto).
Gira nel container spark (pandas e scikit-learn) o nel container nlp.
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from nlp_metrics import (  # noqa: E402
    bootstrap_ci,
    brier,
    calibration_table,
    paired_bootstrap_diff,
    review_rate_table,
    summary,
)
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

NLP = Path(os.getenv("NLP_DIR", "data/nlp"))
REPORTS = Path(os.getenv("REPORTS_DIR", "reports"))
MODELS = Path(os.getenv("MODELS_DIR", "models"))
B = int(os.getenv("BOOTSTRAP_B", "200"))
VOCAB_SAMPLE = int(os.getenv("VOCAB_SAMPLE", "120000"))
MAX_FEATURES = int(os.getenv("MAX_FEATURES", "300000"))
C_GRID = [float(c) for c in os.getenv("C_GRID", "0.3,1,3,10").split(",")]
(REPORTS / "figures").mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

KEYWORDS = [r"injur", r"hospital", r"ambulance", r"emergency room|\ber\b", r"surgery|surgical", r"stitches",
            r"broken|fractur", r"concussion", r"bleed", r"\bburn", r"killed|died|death|fatal",
            r"paramedic|\bems\b", r"medical"]

lines = []


def log(text=""):
    print(text, flush=True)
    lines.append(text)


def block(df, index=False):
    log("```text")
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}", "display.width", 200):
        log(df.to_string(index=index))
    log("```")


def keyword_score(texts: pd.Series) -> np.ndarray:
    total = np.zeros(len(texts), dtype=np.int16)
    for p in KEYWORDS:
        total += texts.str.contains(p, case=False, regex=True).to_numpy().astype(np.int16)
    return total


# ------------------------------------------------------------------ dati
df = pd.read_parquet(NLP / "narratives", columns=["odino", "text", "serious", "split", "n_words"])
tr, va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "validation", "test"))
del df
log("# Milestone 4: baseline lessicale per il triage delle narrative NHTSA")
log()
log(f"Train {len(tr):,} (esiti gravi {tr['serious'].mean():.2%}) | validation {len(va):,} ({va['serious'].mean():.2%}) "
    f"| test {len(te):,} ({te['serious'].mean():.2%}) | bootstrap B={B}")

# ------------------------------------------------------------------ controlli di leakage sul train
log()
log("## 1. Controlli di leakage (train)")
base = tr["serious"].mean()
tr["len_bin"] = pd.qcut(tr["n_words"], 5, duplicates="drop")
by_len = tr.groupby("len_bin", observed=True).agg(reclami=("serious", "size"), tasso_grave=("serious", "mean"))
by_len["lift"] = by_len["tasso_grave"] / base
log()
log("Tasso di esito grave per quintile di lunghezza (una forte dipendenza sarebbe un segnale spurio):")
block(by_len.reset_index().astype({"len_bin": str}))


def edge_table(texts: pd.Series, y: pd.Series, regex: str, min_count: int = 300, top: int = 12):
    tok = texts.str.extract(regex, expand=False).str.lower()
    g = pd.DataFrame({"tok": tok, "y": y}).dropna().groupby("tok").agg(reclami=("y", "size"), tasso_grave=("y", "mean"))
    g = g[g["reclami"] >= min_count]
    g["lift"] = g["tasso_grave"] / base
    both = pd.concat([g.sort_values("lift", ascending=False).head(top), g.sort_values("lift").head(top // 2)])
    return both[~both.index.duplicated()].reset_index()


log()
log("Ultima parola della narrativa: livelli con piu' associazione all'etichetta (min 300 reclami). "
    "Un lift lontano da 1 su una parola 'di forma' (non di contenuto) indica un artefatto da ripulire:")
block(edge_table(tr["text"], tr["serious"], r"(\S+)\s*$"))
log()
log("Prime due parole della narrativa:")
block(edge_table(tr["text"], tr["serious"], r"^(\S+\s+\S+)"))

# ------------------------------------------------------------------ regola a parole chiave
t0 = time.time()
kw = {"val": keyword_score(va["text"]), "test": keyword_score(te["text"])}
print(f"[tempo] regola a parole chiave: {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ TF-IDF + regressione logistica
t0 = time.time()
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=MAX_FEATURES, sublinear_tf=True,
                      dtype=np.float32)
vec.fit(tr["text"].sample(min(VOCAB_SAMPLE, len(tr)), random_state=0))  # il vocabolario si fissa su un campione
X_tr, X_va, X_te = (vec.transform(d["text"]) for d in (tr, va, te))
y_tr, y_va, y_te = tr["serious"].to_numpy(), va["serious"].to_numpy(), te["serious"].to_numpy()
print(f"[tempo] vettorizzazione: {time.time() - t0:.0f}s | vocabolario {len(vec.vocabulary_):,}", flush=True)

grid = []
best = (-1.0, None, None)
for C in C_GRID:
    t0 = time.time()
    clf = LogisticRegression(C=C, solver="liblinear", max_iter=300)
    clf.fit(X_tr, y_tr)
    ap = average_precision_score(y_va, clf.predict_proba(X_va)[:, 1])
    grid.append({"C": C, "val_pr_auc": ap})
    print(f"[tempo] C={C}: {time.time() - t0:.0f}s | PR-AUC validation {ap:.4f}", flush=True)
    if ap > best[0]:
        best = (ap, C, clf)
_, best_c, clf = best
p_va, p_te = clf.predict_proba(X_va)[:, 1], clf.predict_proba(X_te)[:, 1]

log()
log("## 2. Scelta di C (PR-AUC su validation)")
log()
block(pd.DataFrame(grid))
log(f"Scelto C = {best_c}.")

# ------------------------------------------------------------------ risultati
log()
log("## 3. Risultati")
rows = []
for name, s_va, s_te in [("Regola a parole chiave", kw["val"], kw["test"]), ("TF-IDF + regressione logistica", p_va, p_te)]:
    a, b = summary(y_va, s_va), summary(y_te, s_te)
    pr, lo, hi = bootstrap_ci(lambda y, s: average_precision_score(y, s), y_te, s_te, B=B)
    rows.append({"modello": name, "pr_auc_val": a["pr_auc"], "pr_auc_test": pr, "ic_low": lo, "ic_high": hi,
                 "roc_auc_test": b["roc_auc"], "base_rate_test": b["base_rate"]})
res = pd.DataFrame(rows)
log()
log("PR-AUC sul test con IC bootstrap al 95%. Con un tasso di base del "
    f"{y_te.mean():.2%}, una PR-AUC casuale varrebbe circa {y_te.mean():.3f}.")
block(res)
res.to_csv(REPORTS / "m4_baseline_results.csv", index=False)

d, lo, hi = paired_bootstrap_diff(lambda y, s: average_precision_score(y, s), y_te, p_te, kw["test"], B=B)
log()
log(f"Differenza di PR-AUC (TF-IDF - regola a parole chiave) sul test: {d:+.4f} (IC 95%: {lo:+.4f}, {hi:+.4f}).")

log()
log("## 4. Tasso di revisione: se un operatore rilegge solo la quota indicata dei reclami con punteggio piu' alto")
for name, s in [("Regola a parole chiave", kw["test"]), ("TF-IDF + regressione logistica", p_te)]:
    log()
    log(f"{name} (test):")
    block(review_rate_table(y_te, s))

log()
log("## 5. Calibrazione del TF-IDF (test, per decili di probabilita')")
log(f"Brier score: validation {brier(y_va, p_va):.4f} | test {brier(y_te, p_te):.4f} "
    f"(riferimento costante: {brier(y_te, np.full(len(y_te), y_tr.mean())):.4f}).")
cal = calibration_table(y_te, p_te)
block(cal)

# ------------------------------------------------------------------ parole con il coefficiente piu' alto
names = np.array(vec.get_feature_names_out())
coef = clf.coef_[0]
order = np.argsort(coef)
top = pd.DataFrame({
    "piu_associate_al_grave": names[order[::-1][:25]], "coef+": coef[order[::-1][:25]],
    "piu_associate_al_non_grave": names[order[:25]], "coef-": coef[order[:25]],
})
log()
log("## 6. Parole e bigrammi con il coefficiente piu' alto: se compaiono parole di forma o firme, c'e' un artefatto")
block(top)

# ------------------------------------------------------------------ grafico
fig, ax = plt.subplots(figsize=(6.4, 4.6))
for name, s in [("Parole chiave", kw["test"] + np.random.default_rng(0).random(len(y_te)) * 1e-6),
                ("TF-IDF + LR", p_te)]:
    p, r, _ = precision_recall_curve(y_te, s)
    ax.plot(r, p, label=f"{name} (PR-AUC {average_precision_score(y_te, s):.3f})")
ax.axhline(y_te.mean(), ls="--", color="grey", lw=1, label=f"tasso di base {y_te.mean():.3f}")
ax.set_xlabel("recall")
ax.set_ylabel("precisione")
ax.set_title("Curva precisione-recall sul test: baseline")
ax.legend()
fig.tight_layout()
fig.savefig(REPORTS / "figures" / "m4_pr_curve_baseline.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ salvataggi
pred = pd.concat([
    pd.DataFrame({"odino": va["odino"], "split": "validation", "serious": y_va, "score_tfidf": p_va, "score_kw": kw["val"]}),
    pd.DataFrame({"odino": te["odino"], "split": "test", "serious": y_te, "score_tfidf": p_te, "score_kw": kw["test"]}),
])
pred.to_parquet(NLP / "pred_tfidf.parquet", index=False)
joblib.dump({"vectorizer": vec, "classifier": clf}, MODELS / "nlp_tfidf.joblib")
meta = {
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "task": "esito grave (feriti, decessi o cure mediche) da narrativa del reclamo NHTSA",
    "split": "train <= 2023, validation 2024, test >= 2025",
    "vectorizer": {"ngram_range": [1, 2], "min_df": 5, "max_features": MAX_FEATURES, "sublinear_tf": True},
    "classifier": {"type": "LogisticRegression", "solver": "liblinear", "C": best_c},
    "test": res.set_index("modello")[["pr_auc_test", "ic_low", "ic_high", "roc_auc_test"]].round(4).to_dict("index"),
}
(MODELS / "nlp_tfidf_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
(REPORTS / "m4_baseline_tfidf.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nReport scritto in {REPORTS / 'm4_baseline_tfidf.md'}")
