"""Funzioni per l'analisi dei rischi dei modelli (milestone 5): stabilita' della popolazione, calibrazione per gruppo."""
import numpy as np
import pandas as pd

AGE_BINS, AGE_LABELS = [17, 24, 34, 49, 64, 200], ["18-24", "25-34", "35-49", "50-64", "65+"]
VEH_AGE_BINS, VEH_AGE_LABELS = [-1, 0, 2, 5, 10, 15, 200], ["0", "1-2", "3-5", "6-10", "11-15", "16+"]
BM_BINS, BM_LABELS = [0, 50, 75, 100, 125, 1000], ["<=50", "51-75", "76-100", "101-125", "126+"]


def band(values, bins, labels):
    return pd.cut(pd.Series(np.asarray(values)), bins=bins, labels=labels).astype(str).to_numpy()


def psi(reference, new, bins: int = 10, new_weights=None, eps: float = 1e-6) -> float:
    """Population Stability Index tra una distribuzione di riferimento e una nuova (eventualmente pesata).
    Le fasce sono i decili del riferimento. Regola pratica: < 0,10 stabile; 0,10-0,25 spostamento moderato; > 0,25 forte."""
    reference = np.asarray(reference, dtype=float)
    new = np.asarray(new, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    p_ref = np.bincount(np.searchsorted(edges, reference, side="right"), minlength=len(edges) + 1).astype(float)
    p_new = np.bincount(np.searchsorted(edges, new, side="right"), weights=new_weights, minlength=len(edges) + 1).astype(float)
    p_ref, p_new = p_ref / p_ref.sum(), p_new / p_new.sum()
    p_ref, p_new = np.clip(p_ref, eps, None), np.clip(p_new, eps, None)
    return float(np.sum((p_new - p_ref) * np.log(p_new / p_ref)))


def psi_categorical(reference, new, new_weights=None, eps: float = 1e-6) -> float:
    """PSI per una variabile categorica (livelli visti nel riferimento o nel nuovo campione)."""
    levels = sorted(set(pd.unique(np.asarray(reference))) | set(pd.unique(np.asarray(new))))
    ref = pd.Series(np.asarray(reference)).value_counts(normalize=True).reindex(levels).fillna(0).to_numpy()
    w = np.ones(len(new)) if new_weights is None else np.asarray(new_weights, dtype=float)
    nw = pd.Series(w).groupby(np.asarray(new)).sum().reindex(levels).fillna(0).to_numpy()
    nw = nw / nw.sum()
    ref, nw = np.clip(ref, eps, None), np.clip(nw, eps, None)
    return float(np.sum((nw - ref) * np.log(nw / ref)))


def group_calibration(groups, observed, expected: dict, exposure, B: int = 200, seed: int = 0) -> pd.DataFrame:
    """Per ogni gruppo: sinistri osservati contro attesi di ciascun modello, con IC bootstrap al 95% (ricampionando le polizze).
    `expected` = {nome modello: sinistri attesi per polizza}. rapporto = osservati / attesi (1 = calibrato; < 1 = il modello sovrastima)."""
    codes, levels = pd.factorize(np.asarray(groups), sort=True)
    g, n = len(levels), len(codes)
    obs = np.asarray(observed, dtype=float)
    expo = np.asarray(exposure, dtype=float)
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, n, n) for _ in range(B)]

    def sums(values, i=None):
        if i is None:
            return np.bincount(codes, weights=values, minlength=g)
        return np.bincount(codes[i], weights=values[i], minlength=g)

    out = pd.DataFrame({"gruppo": levels, "anni_polizza": sums(expo), "osservati": sums(obs)})
    out["freq_osservata"] = out["osservati"] / out["anni_polizza"]
    obs_b = np.array([sums(obs, i) for i in idx])
    for name, exp in expected.items():
        exp = np.asarray(exp, dtype=float)
        point = sums(obs) / sums(exp)
        boot = obs_b / np.array([sums(exp, i) for i in idx])
        out[f"freq_prevista_{name}"] = sums(exp) / out["anni_polizza"]
        out[f"rapporto_{name}"] = point
        out[f"ic_low_{name}"] = np.nanpercentile(boot, 2.5, axis=0)
        out[f"ic_high_{name}"] = np.nanpercentile(boot, 97.5, axis=0)
    return out


def flag_outside(df: pd.DataFrame, name: str) -> pd.Series:
    """'*' se l'intervallo di confidenza del rapporto osservati/attesi esclude 1."""
    return np.where((df[f"ic_low_{name}"] > 1) | (df[f"ic_high_{name}"] < 1), "*", "")


def claims_needed(shift: float, power_z: float = 2.8) -> float:
    """Sinistri attesi necessari per rilevare uno spostamento relativo `shift` della frequenza (test a 2 code, 95% di
    confidenza, 80% di potenza: (1,96+0,84)^2 / shift^2). Approssimazione di Poisson: l'errore standard del rapporto e' ~1/sqrt(n)."""
    return float((power_z / shift) ** 2)
