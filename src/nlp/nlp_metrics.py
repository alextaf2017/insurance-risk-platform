"""Metriche per la classificazione di un evento raro (esito grave ~2-3% dei reclami).

Con classi cosi' sbilanciate l'accuratezza e' inutile e la ROC-AUC e' ottimista: si usano la PR-AUC
(average precision), recall e precisione a un tasso di revisione fisso (la quota di reclami che un
operatore e' disposto a rileggere) e la calibrazione.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def review_rate_table(y, score, rates=(0.01, 0.02, 0.05, 0.10, 0.20), seed=0):
    """Se si rivede solo la quota `rate` di reclami con punteggio piu' alto: quanti casi gravi si catturano?
    I pareggi si spezzano a caso, per non favorire modelli a punteggi discreti."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(score)), -score))  # primario: punteggio decrescente
    hits = np.cumsum(y[order])
    n, pos = len(y), y.sum()
    rows = []
    for r in rates:
        k = max(1, int(round(r * n)))
        tp = hits[k - 1]
        rows.append({
            "review_rate_%": 100 * r,
            "reviewed": k,
            "recall": tp / pos,
            "precision": tp / k,
            "lift": (tp / k) / (pos / n),
        })
    return pd.DataFrame(rows)


def summary(y, score):
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "base_rate": float(np.mean(y)),
    }


def brier(y, prob):
    return float(brier_score_loss(y, np.clip(prob, 0, 1)))


def bootstrap_ci(metric, y, score, B=200, seed=42):
    """Intervallo di confidenza al 95% (percentili) ricampionando i reclami."""
    y = np.asarray(y)
    score = np.asarray(score)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].sum() == 0:
            continue
        vals.append(metric(y[i], score[i]))
    return float(metric(y, score)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_diff(metric, y, score_a, score_b, B=200, seed=42):
    """Differenza (A - B) di una metrica, con IC bootstrap appaiato: stessi reclami per i due modelli."""
    y = np.asarray(y)
    a, b = np.asarray(score_a), np.asarray(score_b)
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].sum() == 0:
            continue
        diffs.append(metric(y[i], a[i]) - metric(y[i], b[i]))
    return float(metric(y, a) - metric(y, b)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def calibration_table(y, prob, n_bins=10):
    """Probabilita' prevista contro frequenza osservata per decili di probabilita'."""
    y = np.asarray(y)
    prob = np.asarray(prob)
    q = pd.qcut(pd.Series(prob).rank(method="first"), n_bins, labels=False)
    df = pd.DataFrame({"bin": q, "y": y, "p": prob}).groupby("bin").agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))
    return df.reset_index(drop=True)
