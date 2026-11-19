"""Distribuzioni di riferimento (compatte, senza dati grezzi) e PSI per il monitoraggio degli input in produzione.

Il riferimento si costruisce una volta sul train (`export_assets.py`) e si confronta con i valori visti dal servizio.
Numeriche: fasce = decili del riferimento (stessa convenzione di `risk/risk_lib.py`). Categoriche: quote dei livelli piu'
frequenti piu' un livello '__other__' che raccoglie anche i livelli mai visti.
"""
import numpy as np

EPS = 1e-6
OTHER = "__other__"


def numeric_reference(values, bins: int = 10) -> dict:
    v = np.asarray(values, dtype=float)
    edges = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]))
    counts = np.bincount(np.searchsorted(edges, v, side="right"), minlength=len(edges) + 1).astype(float)
    return {"edges": [float(e) for e in edges], "props": [float(p) for p in counts / counts.sum()], "n": int(len(v)),
            "mean": float(v.mean())}


def categorical_reference(values, top: int = 25) -> dict:
    s = np.asarray(values).astype(str)
    levels, counts = np.unique(s, return_counts=True)
    order = np.argsort(-counts)[:top]
    props = {str(levels[i]): float(counts[i] / counts.sum()) for i in order}
    rest = 1.0 - sum(props.values())
    if rest > 1e-9:
        props[OTHER] = float(rest)
    return {"props": props, "n": int(len(s))}


def _psi(p_ref, p_new) -> float:
    p_ref, p_new = np.clip(np.asarray(p_ref, float), EPS, None), np.clip(np.asarray(p_new, float), EPS, None)
    p_ref, p_new = p_ref / p_ref.sum(), p_new / p_new.sum()
    return float(np.sum((p_new - p_ref) * np.log(p_new / p_ref)))


def psi_numeric(ref: dict, values) -> float:
    v = np.asarray(values, dtype=float)
    edges = np.asarray(ref["edges"], dtype=float)
    p_new = np.bincount(np.searchsorted(edges, v, side="right"), minlength=len(edges) + 1).astype(float)
    return _psi(ref["props"], p_new)


def psi_categorical(ref: dict, values) -> float:
    known = ref["props"]
    levels = list(known)
    idx = {k: i for i, k in enumerate(levels)}
    other = idx.get(OTHER)
    counts = np.zeros(len(levels) + (0 if other is not None else 1))
    for x in np.asarray(values).astype(str):
        i = idx.get(x, other if other is not None else len(levels))
        counts[i] += 1
    p_ref = np.array([known[k] for k in levels] + ([] if other is not None else [0.0]))
    return _psi(p_ref, counts)


def status_from_psi(value: float, warn: float = 0.05, action: float = 0.10) -> str:
    """Soglie proposte nella model card dei modelli assicurativi (ipotesi da validare, non derivate dai dati)."""
    return "action" if value >= action else "warn" if value >= warn else "ok"
