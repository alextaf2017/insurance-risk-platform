"""Libreria condivisa per la milestone 3 (modelli tabellari): feature, pipeline e metriche attuariali.

Convenzioni: i modelli di frequenza predicono il TASSO di sinistri per anno-polizza
(target = claim_nb / exposure, peso = exposure). Il numero atteso di sinistri di una polizza e' tasso * exposure.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, SplineTransformer, StandardScaler

# `area` e' escluso di default: ha correlazione 0,976 con `log_density` (milestone 2B).
NUM = ["driv_age_c", "veh_age_c", "bonus_malus_c", "veh_power", "log_density"]
CAT = ["veh_gas", "veh_brand", "region"]
SPLINE_COLS = ["driv_age_c", "veh_age_c", "bonus_malus_c", "log_density"]


# --------------------------------------------------------------------------- pipeline
def glm_preprocessor(num, cat, n_knots=5):
    """Spline cubiche per le variabili continue, scala per le lineari, one-hot per le categoriche.
    I livelli rari (< 1% delle righe) vengono raggruppati automaticamente in una categoria 'infrequent'."""
    spline_cols = [c for c in num if c in SPLINE_COLS]
    linear_cols = [c for c in num if c not in SPLINE_COLS]
    parts = []
    if spline_cols:
        parts.append(("spline", SplineTransformer(n_knots=n_knots, degree=3, extrapolation="constant"), spline_cols))
    if linear_cols:
        parts.append(("linear", StandardScaler(), linear_cols))
    if cat:
        parts.append(("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01), cat))
    return ColumnTransformer(parts)


def make_glm(kind="poisson", num=NUM, cat=CAT, n_knots=5, alpha=1e-4, **kwargs):
    reg = {
        "poisson": lambda: PoissonRegressor(alpha=alpha, max_iter=1000),
        "gamma": lambda: GammaRegressor(alpha=alpha, max_iter=1000),
        "tweedie": lambda: TweedieRegressor(power=kwargs.get("power", 1.9), link="log", alpha=alpha, max_iter=500),
    }[kind]()
    return Pipeline([("prep", glm_preprocessor(num, cat, n_knots)), ("model", reg)])


def make_hgb(loss="poisson", num=NUM, cat=CAT, **params):
    """Gradient boosting con early stopping (10% di validazione interna) e categoriche native."""
    prep = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cat)],
        remainder="passthrough",
    )
    model = HistGradientBoostingRegressor(
        loss=loss,
        categorical_features=[True] * len(cat) + [False] * len(num),
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        max_iter=500,
        random_state=42,
        **params,
    )
    return Pipeline([("prep", prep), ("model", model)])


def frame(df, num=NUM, cat=CAT):
    """Seleziona solo le colonne usate dai modelli, in ordine fisso (le categoriche prima)."""
    return df[cat + num]


def fit_weighted(pipe, X, y, w):
    pipe.fit(X, y, model__sample_weight=w)
    return pipe


# --------------------------------------------------------------------------- metriche di frequenza
def poisson_unit_dev(y, mu):
    """Devianza di Poisson per riga: 2 * (y*log(y/mu) - (y - mu)). Per y = 0 vale 2*mu."""
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-12, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2.0 * (t - (y - mu))


def dev_per_exposure(claims, rate, expo):
    """Devianza di Poisson media per anno-polizza (piu' bassa = meglio)."""
    return float(poisson_unit_dev(claims, rate * expo).sum() / expo.sum())


def lorenz_gini(rate, losses, expo):
    """Gini attuariale: polizze ordinate per rischio previsto crescente, curva di Lorenz di
    esposizione cumulata contro perdita cumulata. 0 = nessuna discriminazione; piu' alto = meglio."""
    order = np.argsort(rate, kind="stable")
    ce = np.r_[0.0, np.cumsum(expo[order])]
    cl = np.r_[0.0, np.cumsum(losses[order])]
    ce, cl = ce / ce[-1], cl / cl[-1]
    auc = np.sum((ce[1:] - ce[:-1]) * (cl[1:] + cl[:-1]) / 2.0)
    return float(1.0 - 2.0 * auc), ce, cl


def decile_table(rate, losses, expo, n=10):
    """Calibrazione per decili di esposizione: perdita osservata vs prevista per anno-polizza."""
    order = np.argsort(rate, kind="stable")
    ce = np.cumsum(expo[order]) / expo.sum()
    bins = np.minimum((ce * n - 1e-12).astype(int), n - 1)
    df = pd.DataFrame({
        "bin": bins,
        "expo": expo[order],
        "obs": losses[order],
        "pred": (rate * expo)[order],
    }).groupby("bin").sum()
    out = pd.DataFrame({
        "decile": df.index + 1,
        "exposure": df["expo"].round(0),
        "observed": (df["obs"] / df["expo"]).values,
        "predicted": (df["pred"] / df["expo"]).values,
    })
    out["obs/pred"] = out["observed"] / out["predicted"]
    return out


def paired_bootstrap(d_a, d_b, expo, B=200, seed=42):
    """Differenza di devianza per anno-polizza (A - B) con IC bootstrap sulle polizze.
    d_a e d_b sono le devianze per riga dei due modelli. Negativo = A migliore."""
    rng = np.random.default_rng(seed)
    n = len(expo)
    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        diffs[b] = (d_a[idx].sum() - d_b[idx].sum()) / expo[idx].sum()
    point = (d_a.sum() - d_b.sum()) / expo.sum()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


# --------------------------------------------------------------------------- metriche di severita'
def gamma_unit_dev(y, mu):
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-12, None)
    return 2.0 * (-np.log(y / mu) + (y - mu) / mu)


def to_text(df: pd.DataFrame) -> str:
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}"):
        return df.to_string(index=False)
