"""Funzioni statistiche per la milestone 2B (indipendenti da Spark, quindi testabili).

Tutte le analisi vanno eseguite sul solo split 'train': il test set non si tocca prima della
valutazione finale dei modelli.
"""
import gc

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Termini del modello di frequenza (nome leggibile -> parte destra della formula patsy).
# I livelli di riferimento sono scelti tra i più popolati, per stime più stabili.
FREQ_TERMS = {
    "area": "C(area)",
    "driv_age_band": 'C(driv_age_band, Treatment(reference="35-49"))',
    "bonus_malus": "bonus_malus_c",
    "veh_age": "veh_age_c",
    "veh_power": "veh_power",
    "veh_gas": "C(veh_gas)",
    "veh_brand": 'C(veh_brand, Treatment(reference="B12"))',
    "log_density": "log_density",
    "region": 'C(region, Treatment(reference="R24"))',
}

# Termini del modello di severità (meno dati: nessuna regione né marca).
SEV_TERMS = {
    "area": "C(area)",
    "driv_age_band": 'C(driv_age_band, Treatment(reference="35-49"))',
    "bonus_malus": "bonus_malus_c",
    "veh_age": "veh_age_c",
    "veh_power": "veh_power",
    "veh_gas": "C(veh_gas)",
    "log_density": "log_density",
}

BM_EDGES = [50, 75, 100, 125]
BM_LABELS = ["1: <=50", "2: 51-75", "3: 76-100", "4: 101-125", "5: 126+"]


def add_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge la fascia di Bonus-Malus e il codice ordinale dell'area (A=1 ... F=6)."""
    out = df.copy()
    idx = np.searchsorted(BM_EDGES, out["bonus_malus_c"].to_numpy(), side="left")
    out["bm_band"] = np.array(BM_LABELS)[idx]
    out["area_code"] = out["area"].astype(str).map({a: i + 1 for i, a in enumerate("ABCDEF")}).astype(float)
    return out


def _fit(df, response, rhs, family, offset=None):
    return smf.glm(f"{response} ~ {rhs}", data=df, family=family, offset=offset).fit()


def drop_one_tests(df, response, terms, family, offset=None, chi2_test=True):
    """Test drop-one: per ogni termine si rifà il modello senza e si confronta con il completo.

    Colonne: df_term, delta_dev (aumento di devianza togliendo il termine), pct_null_dev (effect size),
    p_chi2 (LR con phi=1, solo per Poisson) e p_F (test F quasi-likelihood con phi stimato da Pearson).
    Con n grande i p-value sono quasi sempre ~0: conta l'effect size.
    """
    full = _fit(df, response, " + ".join(terms.values()), family, offset)
    phi = full.pearson_chi2 / full.df_resid
    rows = []
    for name, rhs_term in terms.items():
        rest = " + ".join(t for k, t in terms.items() if k != name) or "1"
        red = _fit(df, response, rest, family, offset)
        d_df = int(red.df_resid - full.df_resid)
        d_dev = float(red.deviance - full.deviance)
        p_f = stats.f.sf((d_dev / d_df) / phi, d_df, full.df_resid)
        del red
        gc.collect()
        rows.append({
            "term": name,
            "df_term": d_df,
            "delta_dev": round(d_dev, 1),
            "pct_null_dev": round(100 * d_dev / full.null_deviance, 3),
            "p_chi2": stats.chi2.sf(d_dev, d_df) if chi2_test else np.nan,
            "p_F": p_f,
        })
    table = pd.DataFrame(rows).sort_values("delta_dev", ascending=False).reset_index(drop=True)
    info = {
        "phi_pearson": float(phi),
        "deviance": float(full.deviance),
        "null_deviance": float(full.null_deviance),
        "n": int(full.nobs),
        "n_params": int(len(full.params)),
    }
    return table, info


def vif_table(df, cols, sample=100_000, seed=42):
    """VIF (con costante) e correlazioni di Pearson e Spearman su un campione."""
    data = df[cols].astype(float)
    if len(data) > sample:
        data = data.sample(sample, random_state=seed)
    X = sm.add_constant(data.to_numpy())
    vif = pd.DataFrame({
        "feature": cols,
        "VIF": [variance_inflation_factor(X, i + 1) for i in range(len(cols))],
    }).round(2)
    return vif, data.corr(method="pearson").round(3), data.corr(method="spearman").round(3)


def bootstrap_relativities(df, by, B=200, seed=42, alpha=0.05):
    """Relatività di frequenza per livello di `by` rispetto al portafoglio, con IC bootstrap
    (ricampionamento delle polizze, percentili)."""
    rng = np.random.default_rng(seed)
    codes, levels = pd.factorize(df[by], sort=True)
    g = len(levels)
    expo = df["exposure"].to_numpy(dtype=float)
    claims = df["claim_nb"].to_numpy(dtype=float)
    n = len(df)

    def rel(idx):
        e = np.bincount(codes[idx], weights=expo[idx], minlength=g)
        c = np.bincount(codes[idx], weights=claims[idx], minlength=g)
        return (c / e) / (c.sum() / e.sum())

    point = rel(np.arange(n))
    boots = np.array([rel(rng.integers(0, n, n)) for _ in range(B)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    n_claims = np.bincount(codes, weights=claims, minlength=g)
    return pd.DataFrame({
        by: levels,
        "claims": n_claims.astype(int),
        "relativity": point.round(3),
        "ci_low": lo.round(3),
        "ci_high": hi.round(3),
        "ci_width": (hi - lo).round(3),
    })


def hill_estimator(x, ks=(50, 100, 200, 400)):
    """Stimatore di Hill dell'indice di coda alpha sui k valori più grandi.

    alpha < 2 indica varianza infinita; alpha < 1 anche media infinita (modello Pareto).
    """
    xs = np.sort(np.asarray(x, dtype=float))[::-1]
    rows = []
    for k in ks:
        if k >= len(xs):
            continue
        hill = np.mean(np.log(xs[:k]) - np.log(xs[k]))
        alpha = 1.0 / hill
        rows.append({
            "k": k,
            "threshold": round(float(xs[k]), 0),
            "alpha": round(alpha, 3),
            "se": round(alpha / np.sqrt(k), 3),
        })
    return pd.DataFrame(rows)


def tail_shares(x, fracs=(0.001, 0.01, 0.05, 0.10)):
    """Quota dell'importo totale detenuta dalla frazione più grande dei sinistri."""
    xs = np.sort(np.asarray(x, dtype=float))[::-1]
    total = xs.sum()
    return pd.DataFrame({
        "top_fraction_%": [100 * f for f in fracs],
        "n_claims": [max(1, int(round(f * len(xs)))) for f in fracs],
        "share_of_total_%": [round(100 * xs[: max(1, int(round(f * len(xs))))].sum() / total, 1) for f in fracs],
    })


def to_text(df: pd.DataFrame) -> str:
    """Tabella in testo allineato, da inserire in un blocco di codice markdown."""
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}"):
        return df.to_string(index=False)
