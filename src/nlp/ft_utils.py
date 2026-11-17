"""Utilita' per il fine-tuning a cascata, senza dipendenze da torch (quindi collaudabili ovunque).

Campionamento pesato: per valutare un modello costoso su un evento raro si tengono TUTTI i positivi e solo una
frazione `q` dei negativi; ogni negativo conta come 1/q reclami nelle metriche. Le stime restano corrette e la
varianza e' molto minore di quella di un campione casuale della stessa dimensione.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def stratified_sample(df: pd.DataFrame, label: str, n_pos: int, n_neg: int, seed: int = 0) -> pd.DataFrame:
    """Campione con n_pos positivi e n_neg negativi (meno se non ce ne sono abbastanza), mescolato."""
    pos, neg = df[df[label] == 1], df[df[label] == 0]
    out = pd.concat([pos.sample(min(n_pos, len(pos)), random_state=seed), neg.sample(min(n_neg, len(neg)), random_state=seed)])
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def weighted_subsample(df: pd.DataFrame, label: str, neg_frac: float, seed: int = 0, max_rows: int = 0) -> pd.DataFrame:
    """Tutti i positivi e una frazione neg_frac dei negativi, con colonna `w` (1 per i positivi, 1/neg_frac per i negativi).
    Con max_rows > 0 il campione si limita (mantenendo i pesi coerenti) per le prove rapide."""
    pos = df[df[label] == 1].copy()
    neg = df[df[label] == 0].sample(frac=neg_frac, random_state=seed).copy()
    pos["w"], neg["w"] = 1.0, 1.0 / neg_frac
    out = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if max_rows and len(out) > max_rows:
        scale = len(out) / max_rows
        out = out.iloc[:max_rows].copy()
        out["w"] = out["w"] * scale  # la riduzione e' casuale: si compensa con i pesi
    return out


def length_grouped_batches(lengths, batch_size: int, mega: int = 50, seed: int = 0):
    """Batch di indici con testi di lunghezza simile (meno padding, quindi meno calcolo su CPU).
    Si mescola, si ordina per lunghezza dentro blocchi di `mega` batch e si mescola l'ordine dei batch."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(lengths))
    lengths = np.asarray(lengths)
    batches = []
    block = batch_size * mega
    for s in range(0, len(idx), block):
        chunk = idx[s:s + block]
        chunk = chunk[np.argsort(-lengths[chunk], kind="stable")]
        batches += [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
    order = rng.permutation(len(batches))
    return [batches[i] for i in order]


def weighted_ap(y, score, w) -> float:
    return float(average_precision_score(y, score, sample_weight=w))


def logit(p, eps: float = 1e-7):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def weighted_review_table(y, score, w, n_total: int, pos_total: int, rates, seed: int = 0) -> pd.DataFrame:
    """Rivedendo la quota `rate` dei reclami (sul totale n_total), quanti casi gravi (su pos_total) si catturano.
    Il campione contiene solo i candidati: vale se rate <= quota di candidati sul totale."""
    y, score, w = np.asarray(y), np.asarray(score, dtype=float), np.asarray(w, dtype=float)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(score)), -score))
    cum_w = np.cumsum(w[order])
    cum_pos = np.cumsum((w * y)[order])
    rows = []
    for r in rates:
        target = r * n_total
        i = min(int(np.searchsorted(cum_w, target)), len(order) - 1)
        tp = cum_pos[i]
        rows.append({"review_rate_%": 100 * r, "reviewed": int(round(target)), "recall": tp / pos_total,
                     "precision": tp / target, "lift": (tp / target) / (pos_total / n_total)})
    return pd.DataFrame(rows)


def recall_at(y, score, w, n_total: int, pos_total: int, rate: float) -> float:
    return float(weighted_review_table(y, score, w, n_total, pos_total, [rate])["recall"].iloc[0])
