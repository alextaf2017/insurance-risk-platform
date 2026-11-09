"""Milestone 3B - Severita', premio puro (frequenza x severita') e benchmark Tweedie.

Scelte di metodo:
- La severita' si modella sull'importo con CAP al percentile CAP_Q del train (la coda pesante, vedi 2B,
  destabilizzerebbe il Gamma); la coda viene reintegrata con un fattore moltiplicativo "tail load" = somma
  degli importi / somma degli importi limitati (train).
- Il modello di severita' si sceglie in CROSS-VALIDATION sul train (non sul test): se non batte la
  severita' costante di almeno l'1%, si usa la costante.
- Il test si usa una sola volta, per la valutazione finale.

Richiede prima 01_frequency.py (usa le previsioni di frequenza sul test).
"""
import json
import os
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from mlib import (  # noqa: E402
    NUM,
    decile_table,
    fit_weighted,
    frame,
    gamma_unit_dev,
    lorenz_gini,
    make_glm,
    make_hgb,
    to_text,
)
from sklearn.model_selection import KFold  # noqa: E402

D = Path(os.getenv("MODELING_DIR", "data/modeling"))
REPORTS = Path(os.getenv("REPORTS_DIR", "reports"))
MODELS = Path(os.getenv("MODELS_DIR", "models"))
CAP_Q = float(os.getenv("CAP_Q", "0.995"))
N_TRIALS = int(os.getenv("N_TRIALS_SEV", "30"))
B = int(os.getenv("BOOTSTRAP_B", "200"))
WITH_TWEEDIE = os.getenv("WITH_TWEEDIE", "1") == "1"
# Quale frequenza usare: claim_nb (default) o sev_claim_count (sinistri con importo, coerente con la severita')
TARGET = os.getenv("FREQ_TARGET", "claim_nb")
SUF = "" if TARGET == "claim_nb" else "_payment"
(REPORTS / "figures").mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEV_NUM, SEV_CAT = NUM, ["veh_gas"]  # pochi sinistri: niente marca e regione

lines = []


def log(text=""):
    print(text, flush=True)
    lines.append(text)


def block(df):
    log("```text")
    log(to_text(df))
    log("```")


# ------------------------------------------------------------------ dati
cl = pd.read_parquet(D / "claims")
pol = pd.read_parquet(D / "policies")
pred = pd.read_parquet(D / f"pred_freq_test{SUF}.parquet")
meta_f = json.loads((MODELS / f"frequency_metadata{SUF}.json").read_text(encoding="utf-8"))
for d in (cl, pol):
    for c in ["veh_gas", "veh_brand", "region", "area"]:
        if c in d:
            d[c] = d[c].astype(str)
    for c in NUM:
        d[c] = d[c].astype("float64")
cl["claim_amount"] = cl["claim_amount"].astype("float64")

ctr = cl[cl["split"] == "train"].reset_index(drop=True)
cte = cl[cl["split"] == "test"].reset_index(drop=True)
ptr = pol[pol["split"] == "train"].reset_index(drop=True)
pte = pol[pol["split"] == "test"].merge(
    pred[["id_pol", "rate_glm", "rate_hgb"]], on="id_pol", how="inner").reset_index(drop=True)
del cl, pol

cap = float(ctr["claim_amount"].quantile(CAP_Q))
y_tr = np.minimum(ctr["claim_amount"].to_numpy(), cap)
y_te = np.minimum(cte["claim_amount"].to_numpy(), cap)
tail_load = float(ctr["claim_amount"].sum() / y_tr.sum())
ones_tr = np.ones(len(ctr))

log(f"# Milestone 3B: severita', premio puro e benchmark Tweedie (frequenza: {TARGET})")
log()
log(f"Sinistri train: {len(ctr):,} | test: {len(cte):,} | cap al percentile {CAP_Q:.3f} del train = {cap:,.0f} | "
    f"tail load = {tail_load:.3f}")
log(f"Importo medio per sinistro (senza cap): train {ctr['claim_amount'].mean():,.0f} | "
    f"test {cte['claim_amount'].mean():,.0f}  <- la differenza mostra quanto la coda destabilizzi le medie")
log()
log("Diagnosi di coerenza tra frequenza e severita':")
log(f"- sinistri contati (claim_nb): train {ptr['claim_nb'].sum():,.0f} | test {pte['claim_nb'].sum():,.0f}")
log(f"- sinistri con importo (tabella severita'): train {len(ctr):,} | test {len(cte):,} "
    f"(rapporto train {len(ctr) / ptr['claim_nb'].sum():.3f} | test {len(cte) / pte['claim_nb'].sum():.3f})")
log("Un rapporto sotto 1 vuol dire che una parte dei sinistri contati non ha un importo registrato: "
    "frequenza (claim_nb) x severita' (solo sinistri con importo) sovrastimerebbe il costo di circa 1/rapporto.")

# ------------------------------------------------------------------ CV sulla severita'
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = list(kf.split(ctr))
Xs_tr = frame(ctr, SEV_NUM, SEV_CAT)


def cv_gamma(make):
    out = []
    for a, b in folds:
        m = fit_weighted(make(), Xs_tr.iloc[a], y_tr[a], ones_tr[a])
        out.append(gamma_unit_dev(y_tr[b], m.predict(Xs_tr.iloc[b])).mean())
    return float(np.mean(out))


def cv_null():
    return float(np.mean([gamma_unit_dev(y_tr[b], np.full(len(b), y_tr[a].mean())).mean() for a, b in folds]))


def study_of(name, objective):
    t0 = time.time()
    s = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(objective, n_trials=N_TRIALS)
    print(f"[tempo] tuning {name}: {time.time() - t0:.0f}s | CV migliore: {s.best_value:.5f}", flush=True)
    return s


glm_s = study_of("Gamma GLM", lambda t: cv_gamma(lambda: make_glm(
    "gamma", SEV_NUM, SEV_CAT, n_knots=t.suggest_int("n_knots", 4, 7), alpha=t.suggest_float("alpha", 1e-4, 1.0, log=True))))


def hgb_obj(t):
    p = {
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 4, 16),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 100, 800, log=True),
        "l2_regularization": t.suggest_float("l2_regularization", 0.1, 50.0, log=True),
    }
    return cv_gamma(lambda: make_hgb("gamma", SEV_NUM, SEV_CAT, **p))


hgb_s = study_of("HGB gamma", hgb_obj)
null_cv = cv_null()
cvs = pd.DataFrame({
    "model": ["Costante", "Gamma GLM (tuned)", "HGB gamma (tuned)"],
    "cv_gamma_dev": [null_cv, glm_s.best_value, hgb_s.best_value],
})
cvs["vs_costante_%"] = 100 * (1 - cvs["cv_gamma_dev"] / null_cv)
log()
log("## 1. Scelta del modello di severita' in cross-validation (5 gruppi sul train, importi con cap)")
log()
block(cvs)

best_cv_model = "glm" if glm_s.best_value <= hgb_s.best_value else "hgb"
best_gain = 100 * (1 - min(glm_s.best_value, hgb_s.best_value) / null_cv)
use_model = best_gain >= 1.0
chosen = best_cv_model if use_model else "constant"
log()
log(f"Regola: si usa un modello solo se migliora la devianza CV di almeno l'1%. "
    f"Migliore: {best_cv_model.upper()} con {best_gain:+.2f}%. **Scelta: {chosen}**.")

# ------------------------------------------------------------------ sensibilita' al cap
rows = []
for q in (0.99, 0.995, 0.999):
    cq = float(ctr["claim_amount"].quantile(q))
    yq = np.minimum(ctr["claim_amount"].to_numpy(), cq)
    gains = []
    for a, b in folds:
        m = fit_weighted(make_glm("gamma", SEV_NUM, SEV_CAT, **glm_s.best_params), Xs_tr.iloc[a], yq[a], ones_tr[a])
        d_m = gamma_unit_dev(yq[b], m.predict(Xs_tr.iloc[b])).mean()
        d_0 = gamma_unit_dev(yq[b], np.full(len(b), yq[a].mean())).mean()
        gains.append(100 * (1 - d_m / d_0))
    rows.append({"cap_quantile": q, "cap_value": round(cq), "tail_load": ctr["claim_amount"].sum() / yq.sum(),
                 "GLM_gain_vs_const_%": float(np.mean(gains))})
log()
log("## 2. Sensibilita' al cap: guadagno CV del Gamma GLM sulla costante, per soglia")
log()
block(pd.DataFrame(rows))
log("Se il guadagno cambia molto con la soglia, la segmentazione della severita' dipende dalle poche perdite "
    "estreme e non e' affidabile.")

# ------------------------------------------------------------------ modello finale di severita' e test
final = None
sev_const = float(y_tr.mean())
if use_model:
    params = glm_s.best_params if chosen == "glm" else hgb_s.best_params
    final = (make_glm("gamma", SEV_NUM, SEV_CAT, **params) if chosen == "glm"
             else make_hgb("gamma", SEV_NUM, SEV_CAT, **params))
    fit_weighted(final, Xs_tr, y_tr, ones_tr)
    sev_te_claims = final.predict(frame(cte, SEV_NUM, SEV_CAT))
    sev_te_pol = final.predict(frame(pte, SEV_NUM, SEV_CAT))
else:
    sev_te_claims = np.full(len(cte), sev_const)
    sev_te_pol = np.full(len(pte), sev_const)

d_model = gamma_unit_dev(y_te, sev_te_claims)
d_null = gamma_unit_dev(y_te, np.full(len(cte), sev_const))
rng = np.random.default_rng(42)
boots = np.array([(lambda i: d_null[i].mean() - d_model[i].mean())(rng.integers(0, len(cte), len(cte)))
                  for _ in range(B)])
log()
log("## 3. Severita' sul test (sinistri con cap)")
log()
log(f"Devianza Gamma media: costante {d_null.mean():.4f} | modello scelto ({chosen}) {d_model.mean():.4f} | "
    f"guadagno {d_null.mean() - d_model.mean():+.4f} (IC bootstrap 95%: "
    f"{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}).")

# ------------------------------------------------------------------ premio puro sul test
expo = pte["exposure"].to_numpy()
loss = pte["total_claim_amount"].to_numpy()  # perdita reale, SENZA cap
cap_by_pol = pd.Series(y_te).groupby(cte["id_pol"].to_numpy()).sum()  # perdita con cap per sinistro
loss_cap = pte["id_pol"].map(cap_by_pol).fillna(0.0).to_numpy()
variants = {
    "freq GLM x sev": pte["rate_glm"].to_numpy() * expo * sev_te_pol * tail_load,
    "freq HGB x sev": pte["rate_hgb"].to_numpy() * expo * sev_te_pol * tail_load,
}

tails = {name: tail_load for name in variants}
if WITH_TWEEDIE:
    t0 = time.time()
    loss_tr = ptr["total_claim_amount"].to_numpy()
    capped_tr = np.minimum(loss_tr, cap)
    tail_pol = float(loss_tr.sum() / capped_tr.sum())
    expo_tr = ptr["exposure"].to_numpy()
    tw = make_glm("tweedie", n_knots=meta_f["glm_params"]["n_knots"], alpha=meta_f["glm_params"]["alpha"], power=1.9)
    fit_weighted(tw, frame(ptr), capped_tr / expo_tr, expo_tr)
    variants["Tweedie (1.9) diretto"] = tw.predict(frame(pte)) * expo * tail_pol
    tails["Tweedie (1.9) diretto"] = tail_pol
    print(f"[tempo] Tweedie: {time.time() - t0:.0f}s", flush=True)

rows = []
for name, pp in variants.items():
    ratios, ratios_cap = [], []
    pp_cap = pp / tails[name]  # previsione senza il tail load: confrontabile con la perdita con cap
    for _ in range(B):
        i = rng.integers(0, len(pte), len(pte))
        ratios.append(loss[i].sum() / pp[i].sum())
        ratios_cap.append(loss_cap[i].sum() / pp_cap[i].sum())
    rows.append({
        "model": name,
        "actual_total": loss.sum(),
        "predicted_total": pp.sum(),
        "actual/pred": loss.sum() / pp.sum(),
        "ci_low": np.percentile(ratios, 2.5),
        "ci_high": np.percentile(ratios, 97.5),
        "capped_act/pred": loss_cap.sum() / pp_cap.sum(),
        "cap_ci_low": np.percentile(ratios_cap, 2.5),
        "cap_ci_high": np.percentile(ratios_cap, 97.5),
        "gini_loss": lorenz_gini(pp / expo, loss, expo)[0],
    })
pp_res = pd.DataFrame(rows)
log()
log("## 4. Premio puro sul test: perdita reale (senza cap) contro prevista")
log()
log("`actual/pred` vicino a 1 = portafoglio ben calibrato; con code pesanti l'IC bootstrap e' troppo stretto, "
    "quindi si guarda anche `capped_act/pred` (perdita con cap per sinistro), molto meno rumoroso. "
    "`gini_loss` = capacita' di ordinare le polizze per costo atteso.")
block(pp_res)
pp_res.to_csv(REPORTS / f"m3_pricing_results{SUF}.csv", index=False)

best_pp = pp_res.sort_values("gini_loss", ascending=False).iloc[0]["model"]
cal = decile_table(variants[best_pp] / expo, loss, expo)
log()
log(f"## 5. Calibrazione per decili del premio puro ({best_pp}): perdita osservata vs prevista per anno-polizza")
log()
block(cal)
cal.to_csv(REPORTS / f"m3_pricing_calibration{SUF}.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for name, pp in variants.items():
    t = decile_table(pp / expo, loss, expo)
    axes[0].plot(t["predicted"], t["observed"], "o-", label=name)
    g, ce, cl_ = lorenz_gini(pp / expo, loss, expo)
    axes[1].plot(ce, cl_, label=f"{name}: Gini {g:.3f}")
lim = [0, max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])]
axes[0].plot(lim, lim, "--", color="grey", lw=1)
axes[0].set_xlabel("perdita prevista per anno-polizza")
axes[0].set_ylabel("perdita osservata")
axes[0].set_title("Premio puro: calibrazione per decili (test)")
axes[0].legend(fontsize=8)
axes[1].plot([0, 1], [0, 1], "--", color="grey", lw=1)
axes[1].set_xlabel("esposizione cumulata (costo atteso crescente)")
axes[1].set_ylabel("perdita cumulata")
axes[1].set_title("Curva di Lorenz sulla perdita (test)")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(REPORTS / "figures" / f"m3_pricing_calibration_lorenz{SUF}.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ salvataggi
if final is not None:
    joblib.dump(final, MODELS / "severity.joblib")
meta = {
    "cap_quantile": CAP_Q,
    "cap_value": cap,
    "tail_load": tail_load,
    "frequency_target": TARGET,
    "severity_model": chosen,
    "severity_constant_capped": sev_const,
    "severity_features": {"numeric": SEV_NUM, "categorical": SEV_CAT},
    "glm_params": glm_s.best_params,
    "hgb_params": hgb_s.best_params,
    "pricing_test": pp_res.set_index("model")[["actual/pred", "gini_loss"]].round(4).to_dict("index"),
}
(MODELS / f"pricing_metadata{SUF}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
(REPORTS / f"m3_severity_pricing{SUF}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nReport scritto in {REPORTS / 'm3_severity_pricing.md'}")
