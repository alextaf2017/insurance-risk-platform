"""Milestone 4, passo 3: esperimenti con il TF-IDF per progettare il fine-tuning su CPU.

A) Curva di apprendimento: PR-AUC sul test naturale intero, in funzione del numero di esempi di addestramento
   (campioni con il 25% di positivi, come si faria' per il transformer). Risponde a: quanti dati servono al lessico
   per raggiungere un dato livello? Il C si sceglie sulla validation per ogni dimensione.
B) Troncamento: PR-AUC con il testo intero, le prime 96 parole, testa+coda (48+48) e le prime 192 parole.
   Un transformer con 128 token vede circa 96 parole: la perdita misura quanto segnale si butta via.

Output: reports/m4_tfidf_experiments.md, reports/figures/m4_learning_curve.png, reports/m4_learning_curve.csv.
Variabili: LC_SIZES, POS_FRAC (0.25), TRUNC_SAMPLE (150000), BOOTSTRAP_B (100).
"""
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from nlp_metrics import bootstrap_ci  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

NLP = Path(os.getenv("NLP_DIR", "data/nlp"))
REPORTS = Path(os.getenv("REPORTS_DIR", "reports"))
SIZES = [int(x) for x in os.getenv("LC_SIZES", "1000,2500,5000,10000,20000,40000,80000").split(",")]
POS_FRAC = float(os.getenv("POS_FRAC", "0.25"))
TRUNC_SAMPLE = int(os.getenv("TRUNC_SAMPLE", "150000"))
B = int(os.getenv("BOOTSTRAP_B", "100"))
C_GRID = [1.0, 3.0, 10.0]
(REPORTS / "figures").mkdir(parents=True, exist_ok=True)
lines = []


def log(text=""):
    print(text, flush=True)
    lines.append(text)


def block(df):
    log("```text")
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}", "display.width", 200):
        log(df.to_string(index=False))
    log("```")


def ap(y, s):
    return average_precision_score(y, s)


def fit_eval(train_text, y_train, val_text, y_val, test_text, y_test, min_df, c_grid=C_GRID, max_features=300_000):
    """Vettorizza, sceglie C sulla validation e restituisce (PR-AUC val, PR-AUC test con IC, C scelto, vocabolario)."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_features=max_features, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(train_text)
    Xv, Xt = vec.transform(val_text), vec.transform(test_text)
    best = (-1.0, None, None)
    for C in c_grid:
        clf = LogisticRegression(C=C, solver="liblinear", max_iter=300).fit(X, y_train)
        a = ap(y_val, clf.predict_proba(Xv)[:, 1])
        if a > best[0]:
            best = (a, C, clf)
    a_val, C, clf = best
    s_test = clf.predict_proba(Xt)[:, 1]
    point, lo, hi = bootstrap_ci(ap, y_test, s_test, B=B)
    return a_val, point, lo, hi, C, len(vec.vocabulary_)


df = pd.read_parquet(NLP / "narratives", columns=["text", "serious", "split", "n_words"])
tr, va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "validation", "test"))
del df
y_tr, y_va, y_te = tr["serious"].to_numpy(), va["serious"].to_numpy(), te["serious"].to_numpy()
log("# Milestone 4: esperimenti con il TF-IDF per progettare il fine-tuning")
log()
log(f"Train {len(tr):,} ({y_tr.sum():,} positivi) | validation {len(va):,} | test {len(te):,} ({y_te.sum():,} positivi, "
    f"tasso {y_te.mean():.2%}). PR-AUC casuale sul test: {y_te.mean():.3f}.")

# ------------------------------------------------------------------ A) curva di apprendimento
rng = np.random.default_rng(0)
pos_idx, neg_idx = rng.permutation(np.where(y_tr == 1)[0]), rng.permutation(np.where(y_tr == 0)[0])
rows = []
for n in SIZES:
    n_pos = min(int(round(n * POS_FRAC)), len(pos_idx))
    idx = np.concatenate([pos_idx[:n_pos], neg_idx[: n - n_pos]])  # campioni annidati: ogni dimensione contiene la precedente
    t0 = time.time()
    a_val, a_te, lo, hi, C, vocab = fit_eval(
        tr["text"].iloc[idx], y_tr[idx], va["text"], y_va, te["text"], y_te, min_df=2 if n < 20_000 else 5)
    rows.append({"esempi": n, "positivi": n_pos, "pr_auc_val": a_val, "pr_auc_test": a_te, "ic_low": lo, "ic_high": hi,
                 "C": C, "vocabolario": vocab})
    print(f"[tempo] n={n:,}: {time.time() - t0:.0f}s | PR-AUC test {a_te:.4f}", flush=True)

t0 = time.time()
a_val, a_te, lo, hi, C, vocab = fit_eval(tr["text"], y_tr, va["text"], y_va, te["text"], y_te, min_df=5)
rows.append({"esempi": len(tr), "positivi": int(y_tr.sum()), "pr_auc_val": a_val, "pr_auc_test": a_te, "ic_low": lo,
             "ic_high": hi, "C": C, "vocabolario": vocab})
print(f"[tempo] train naturale completo: {time.time() - t0:.0f}s | PR-AUC test {a_te:.4f}", flush=True)
lc = pd.DataFrame(rows)

log()
log("## A. Curva di apprendimento del TF-IDF (campioni con il 25% di positivi; ultima riga: train naturale completo)")
log()
block(lc)
lc.to_csv(REPORTS / "m4_learning_curve.csv", index=False)

fig, ax = plt.subplots(figsize=(6.6, 4.4))
part = lc.iloc[:-1]
ax.errorbar(part["esempi"], part["pr_auc_test"], yerr=[part["pr_auc_test"] - part["ic_low"], part["ic_high"] - part["pr_auc_test"]],
            fmt="o-", capsize=3, label="TF-IDF (campioni arricchiti)")
ax.axhline(lc["pr_auc_test"].iloc[-1], ls="--", color="grey", lw=1, label=f"train completo ({len(tr):,}): {lc['pr_auc_test'].iloc[-1]:.3f}")
ax.axhline(y_te.mean(), ls=":", color="red", lw=1, label="tasso di base")
ax.set_xscale("log")
ax.set_xlabel("esempi di addestramento")
ax.set_ylabel("PR-AUC sul test")
ax.set_title("Curva di apprendimento del TF-IDF")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(REPORTS / "figures" / "m4_learning_curve.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ B) troncamento


def first_n(t, n):
    return " ".join(t.split()[:n])


def head_tail(t, n):
    w = t.split()
    return t if len(w) <= n else " ".join(w[: n // 2] + w[-(n // 2):])


sub = tr.sample(min(TRUNC_SAMPLE, len(tr)), random_state=1)
ys = sub["serious"].to_numpy()
variants = {
    "testo intero": lambda t: t,
    "prime 96 parole (~128 token)": lambda t: first_n(t, 96),
    "testa+coda 48+48 parole": lambda t: head_tail(t, 96),
    "prime 192 parole (~256 token)": lambda t: first_n(t, 192),
}
rows = []
for name, f in variants.items():
    t0 = time.time()
    a_val, a_te, lo, hi, C, _ = fit_eval(
        sub["text"].map(f), ys, va["text"].map(f), y_va, te["text"].map(f), y_te, min_df=5, c_grid=[3.0])
    kept = float(np.mean([min(len(t.split()), len(f(t).split())) / max(1, len(t.split())) for t in sub["text"].iloc[:5000]]))
    rows.append({"variante": name, "parole_conservate_%": 100 * kept, "pr_auc_val": a_val, "pr_auc_test": a_te,
                 "ic_low": lo, "ic_high": hi})
    print(f"[tempo] {name}: {time.time() - t0:.0f}s | PR-AUC test {a_te:.4f}", flush=True)
tab = pd.DataFrame(rows)
log()
log(f"## B. Troncamento del testo (TF-IDF su {len(sub):,} reclami di train, C=3): quanto segnale si perde")
log()
block(tab)
log()
log("Se 'prime 96 parole' perde molto rispetto al testo intero, la coda delle narrative contiene segnale (spesso e' li' "
    "che si dice chi e' rimasto ferito) e conviene la strategia testa+coda o una lunghezza maggiore.")
tab.to_csv(REPORTS / "m4_truncation.csv", index=False)

(REPORTS / "m4_tfidf_experiments.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nReport scritto in {REPORTS / 'm4_tfidf_experiments.md'}")
