"""Milestone 2B - Inferenza statistica sul solo split TRAIN della tabella gold.

1. GLM di Poisson (offset log-esposizione): dispersione e test drop-one per feature
2. Multicollinearità: VIF e correlazioni
3. Incertezza: relatività con intervalli di confidenza bootstrap
4. Severità: coda pesante (Hill), Gamma GLM con e senza cap dei sinistri grandi

Output in reports/: stats_summary.md, tables/*.csv, figures/*.png
"""
import gc
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from stats_lib import (  # noqa: E402
    FREQ_TERMS,
    SEV_TERMS,
    add_bands,
    bootstrap_relativities,
    drop_one_tests,
    hill_estimator,
    tail_shares,
    to_text,
    vif_table,
)

IN = Path(os.getenv("ANALYSIS_DIR", "data/analysis"))
OUT = Path(os.getenv("REPORTS_DIR", "reports"))
B = int(os.getenv("BOOTSTRAP_B", "200"))
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)

lines = []


def log(text=""):
    print(text)
    lines.append(text)


def block(df):
    log("```text")
    log(to_text(df))
    log("```")


def timer(label, t0):
    print(f"[tempo] {label}: {time.time() - t0:.1f}s")


# ------------------------------------------------------------------ dati (esportati da 00_export_train.py)
CAT = ["area", "driv_age_band", "veh_gas", "veh_brand", "region"]

freq = pd.read_parquet(IN / "policies_train").drop(columns=["id_pol"])
for c in freq.columns:
    if c not in CAT:
        freq[c] = freq[c].astype("float64")
freq = add_bands(freq)
for c in CAT:
    freq[c] = freq[c].astype("category")  # molto meno memoria delle stringhe

sev = pd.read_parquet(IN / "claims_train")
for c in ["claim_amount", "bonus_malus_c", "veh_age_c", "veh_power", "log_density"]:
    sev[c] = sev[c].astype("float64")
gc.collect()

log("# Milestone 2B: inferenza statistica (solo split train)")
log()
log(f"Polizze train: {len(freq):,} | sinistri con importo (train): {len(sev):,} | bootstrap B={B}")

# ------------------------------------------------------------------ 1. frequenza
t0 = time.time()
log()
log("## 1. Frequenza: GLM di Poisson con offset log(exposure)")
offset = np.log(freq["exposure"].to_numpy())
f_tab, f_info = drop_one_tests(freq, "claim_nb", FREQ_TERMS, sm.families.Poisson(), offset=offset)
log()
log(f"Dispersione di Pearson (phi): {f_info['phi_pearson']:.3f}  (1 = Poisson puro; > 1 sovradispersione)")
log(f"Devianza residua: {f_info['deviance']:,.0f} | devianza nulla: {f_info['null_deviance']:,.0f} "
    f"| parametri: {f_info['n_params']}")
log()
log("Test drop-one: aumento di devianza togliendo il termine. `pct_null_dev` è l'effect size; "
    "con n grande i p-value sono quasi sempre ~0, quindi conta l'ordine di grandezza dell'effetto.")
block(f_tab)
f_tab.to_csv(OUT / "tables" / "freq_drop_one.csv", index=False)
timer("GLM frequenza e drop-one", t0)

# ------------------------------------------------------------------ 2. multicollinearità
log()
log("## 2. Multicollinearità")
vif_cols = ["bonus_malus_c", "veh_age_c", "driv_age_c", "veh_power", "log_density", "area_code"]
vif, corr_p, corr_s = vif_table(freq, vif_cols)
log()
log("VIF (campione di 100.000 polizze; regola pratica: > 5 attenzione, > 10 problema):")
block(vif)
log()
log("Correlazione di Spearman:")
block(corr_s.reset_index().rename(columns={"index": ""}))
vif.to_csv(OUT / "tables" / "vif.csv", index=False)
corr_s.to_csv(OUT / "tables" / "corr_spearman.csv")

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr_s.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(vif_cols)), vif_cols, rotation=45, ha="right")
ax.set_yticks(range(len(vif_cols)), vif_cols)
for i in range(len(vif_cols)):
    for j in range(len(vif_cols)):
        ax.text(j, i, f"{corr_s.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Correlazione di Spearman (train)")
fig.tight_layout()
fig.savefig(OUT / "figures" / "correlation_heatmap.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ 3. bootstrap
t0 = time.time()
log()
log("## 3. Relatività di frequenza con intervalli di confidenza bootstrap al 95%")
boot = {}
for by in ["driv_age_band", "bm_band", "area"]:
    boot[by] = bootstrap_relativities(freq, by, B=B)
    log()
    log(f"Per `{by}` (relatività = frequenza del livello / frequenza del portafoglio):")
    block(boot[by])
    boot[by].to_csv(OUT / "tables" / f"bootstrap_{by}.csv", index=False)
timer("bootstrap", t0)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, (by, t) in zip(axes, boot.items()):
    x = np.arange(len(t))
    err = np.vstack([t["relativity"] - t["ci_low"], t["ci_high"] - t["relativity"]])
    ax.errorbar(x, t["relativity"], yerr=err, fmt="o", capsize=4)
    ax.axhline(1.0, color="grey", ls="--", lw=1)
    ax.set_xticks(x, t[by].astype(str), rotation=30)
    ax.set_title(by)
    ax.set_ylabel("relatività")
fig.suptitle("Relatività di frequenza con IC bootstrap 95% (train)")
fig.tight_layout()
fig.savefig(OUT / "figures" / "relativities_bootstrap.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ 4. severità
t0 = time.time()
log()
log("## 4. Severità")
log()
log("Quota dell'importo totale detenuta dai sinistri più grandi:")
shares = tail_shares(sev["claim_amount"])
block(shares)
log()
log("Stimatore di Hill dell'indice di coda alpha (alpha < 2: varianza infinita; < 1: anche media infinita). "
    "Una stima stabile al variare di k indica una coda di tipo Pareto.")
hill = hill_estimator(sev["claim_amount"])
block(hill)
shares.to_csv(OUT / "tables" / "tail_shares.csv", index=False)
hill.to_csv(OUT / "tables" / "hill.csv", index=False)

# Gamma con link log. La risposta è riscalata per la media: la deviance e il test F sono invarianti
# alla scala, ma l'IRLS converge in modo molto più stabile.
gamma = sm.families.Gamma(sm.families.links.Log())
cap = float(sev["claim_amount"].quantile(0.995))
sev["y"] = sev["claim_amount"] / sev["claim_amount"].mean()
capped = np.minimum(sev["claim_amount"], cap)
sev["y_cap"] = capped / capped.mean()

s_all, s_all_info = drop_one_tests(sev, "y", SEV_TERMS, gamma, chi2_test=False)
s_cap, s_cap_info = drop_one_tests(sev, "y_cap", SEV_TERMS, gamma, chi2_test=False)
cmp = s_all[["term", "pct_null_dev", "p_F"]].merge(
    s_cap[["term", "pct_null_dev", "p_F"]], on="term", suffixes=("_all", "_cap")
)
log()
log(f"Gamma GLM (link log), test F quasi-likelihood. phi senza cap: {s_all_info['phi_pearson']:.1f}; "
    f"con cap al 99,5° percentile ({cap:,.0f}): {s_cap_info['phi_pearson']:.1f}.")
log("Un effetto che compare solo senza cap è probabilmente guidato da pochi sinistri enormi.")
block(cmp)
cmp.to_csv(OUT / "tables" / "severity_drop_one.csv", index=False)
timer("severità", t0)

# ------------------------------------------------------------------ report
(OUT / "stats_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nReport scritto in {OUT / 'stats_summary.md'}")
