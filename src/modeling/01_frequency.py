"""Milestone 3A - Modelli di frequenza: GLM di Poisson (spline) vs gradient boosting (Poisson).

Flusso: baseline costante -> modelli con parametri di default -> tuning con Optuna (CV a 3 gruppi sui
fold del lake, su un sottocampione del train) -> riadattamento sul train completo -> valutazione UNA volta
sul test. Metriche: devianza di Poisson per anno-polizza, Gini attuariale, calibrazione per decili.

Output: reports/m3_frequency.md, reports/figures/m3_*.png, models/frequency_*.joblib/json,
data/modeling/pred_freq_test.parquet (usato dallo script di severita').
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
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlib import (  # noqa: E402
    CAT,
    NUM,
    decile_table,
    dev_per_exposure,
    fit_weighted,
    frame,
    lorenz_gini,
    make_glm,
    make_hgb,
    paired_bootstrap,
    poisson_unit_dev,
    to_text,
)
from sklearn.inspection import permutation_importance  # noqa: E402

D = Path(os.getenv("MODELING_DIR", "data/modeling"))
REPORTS = Path(os.getenv("REPORTS_DIR", "reports"))
MODELS = Path(os.getenv("MODELS_DIR", "models"))
N_TRIALS = int(os.getenv("N_TRIALS", "25"))
TUNE_FRAC = float(os.getenv("TUNE_FRAC", "0.5"))
B = int(os.getenv("BOOTSTRAP_B", "200"))
# Target di frequenza: claim_nb (tutti i sinistri contati) oppure sev_claim_count (solo sinistri con importo)
TARGET = os.getenv("FREQ_TARGET", "claim_nb")
assert TARGET in ("claim_nb", "sev_claim_count"), "FREQ_TARGET deve essere claim_nb o sev_claim_count"
SUF = "" if TARGET == "claim_nb" else "_payment"
for p in (REPORTS / "figures", MODELS):
    p.mkdir(parents=True, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

lines = []


def log(text=""):
    print(text, flush=True)
    lines.append(text)


def block(df):
    log("```text")
    log(to_text(df))
    log("```")


# ------------------------------------------------------------------ dati
df = pd.read_parquet(D / "policies")
for c in CAT + ["area"]:
    df[c] = df[c].astype(str)
for c in ["claim_nb", "sev_claim_count", "exposure", "total_claim_amount"] + NUM:
    df[c] = df[c].astype("float64")
tr = df[df["split"] == "train"].reset_index(drop=True)
te = df[df["split"] == "test"].reset_index(drop=True)
del df

claims_tr, expo_tr = tr[TARGET].to_numpy(), tr["exposure"].to_numpy()
claims_te, expo_te = te[TARGET].to_numpy(), te["exposure"].to_numpy()
y_tr = claims_tr / expo_tr
rate_null = claims_tr.sum() / expo_tr.sum()

log(f"# Milestone 3A: modelli di frequenza (target: {TARGET})")
log()
log(f"Train: {len(tr):,} polizze | test: {len(te):,} polizze | trial Optuna: {N_TRIALS} | "
    f"bootstrap B={B} | frequenza di riferimento (train): {rate_null:.4f}")

# ------------------------------------------------------------------ tuning
rng = np.random.default_rng(42)
tune = tr[rng.random(len(tr)) < TUNE_FRAC].reset_index(drop=True)
tune_claims, tune_expo = tune[TARGET].to_numpy(), tune["exposure"].to_numpy()
tune_y = tune_claims / tune_expo
cv_group = tune["fold"].to_numpy() % 3  # 3 gruppi dai fold dell'hash: nessun leakage tra gruppi


def cv_deviance(make, num=NUM, cat=CAT):
    scores = []
    for g in range(3):
        a, b = cv_group != g, cv_group == g
        m = fit_weighted(make(), frame(tune[a], num, cat), tune_y[a], tune_expo[a])
        rate = m.predict(frame(tune[b], num, cat))
        scores.append(dev_per_exposure(tune_claims[b], rate, tune_expo[b]))
    return float(np.mean(scores))


def run_study(name, objective):
    t0 = time.time()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)
    print(f"[tempo] tuning {name}: {time.time() - t0:.0f}s | CV migliore: {study.best_value:.5f}", flush=True)
    return study


glm_study = run_study("GLM", lambda t: cv_deviance(lambda: make_glm(
    "poisson", n_knots=t.suggest_int("n_knots", 4, 8), alpha=t.suggest_float("alpha", 1e-6, 1e-2, log=True))))


def hgb_objective(t):
    params = {
        "learning_rate": t.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 8, 48),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 50, 1000, log=True),
        "l2_regularization": t.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
    }
    return cv_deviance(lambda: make_hgb("poisson", **params))


hgb_study = run_study("HGB", hgb_objective)

# ------------------------------------------------------------------ modelli finali (fit sul train completo)
X_tr, X_te = frame(tr), frame(te)
results, rates_te, rates_tr, fitted = [], {}, {}, {}


def fit_eval(name, model, num=NUM, cat=CAT):
    t0 = time.time()
    Xa, Xb = frame(tr, num, cat), frame(te, num, cat)
    fit_weighted(model, Xa, y_tr, expo_tr)
    r_tr, r_te = model.predict(Xa), model.predict(Xb)
    rates_tr[name], rates_te[name], fitted[name] = r_tr, r_te, model
    print(f"[tempo] {name}: {time.time() - t0:.0f}s", flush=True)


gp, hp = glm_study.best_params, hgb_study.best_params
fit_eval("GLM (default)", make_glm("poisson"))
fit_eval("GLM (tuned)", make_glm("poisson", **gp))
fit_eval("GLM (area al posto di log_density)",
         make_glm("poisson", num=[c for c in NUM if c != "log_density"], cat=CAT + ["area"], **gp),
         num=[c for c in NUM if c != "log_density"], cat=CAT + ["area"])
fit_eval("HGB (default)", make_hgb("poisson"))
fit_eval("HGB (tuned)", make_hgb("poisson", **hp))

rates_tr["Null"] = np.full(len(tr), rate_null)
rates_te["Null"] = np.full(len(te), rate_null)
dev_null_te = dev_per_exposure(claims_te, rates_te["Null"], expo_te)

for name in ["Null", "GLM (default)", "GLM (tuned)", "GLM (area al posto di log_density)", "HGB (default)", "HGB (tuned)"]:
    d_tr = dev_per_exposure(claims_tr, rates_tr[name], expo_tr)
    d_te = dev_per_exposure(claims_te, rates_te[name], expo_te)
    results.append({
        "model": name,
        "dev_train": d_tr,
        "dev_test": d_te,
        "gap_%": 100 * (d_te - d_tr) / d_tr,
        "D2_test_%": 100 * (1 - d_te / dev_null_te),
        # Il modello costante non ordina le polizze (punteggi tutti uguali): Gini 0 per definizione
        "gini_test": 0.0 if name == "Null" else lorenz_gini(rates_te[name], claims_te, expo_te)[0],
    })
res = pd.DataFrame(results)

log()
log("## Risultati sul test (devianza di Poisson per anno-polizza: piu' bassa = meglio)")
log()
log("`gap_%` = degrado da train a test (indizio di overfitting); `D2_test_%` = quota di devianza spiegata "
    "rispetto al modello costante; `gini_test` = potere discriminante (0 = nessuno).")
block(res)
res.to_csv(REPORTS / f"m3_frequency_results{SUF}.csv", index=False)

log()
log("## Parametri scelti da Optuna (CV a 3 gruppi su un sottocampione del train)")
log()
log(f"- GLM: {json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in gp.items()})} "
    f"(CV: {glm_study.best_value:.5f})")
log(f"- HGB: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in hp.items()})} "
    f"(CV: {hgb_study.best_value:.5f})")

# ------------------------------------------------------------------ confronto GLM vs HGB (bootstrap appaiato)
d_glm = poisson_unit_dev(claims_te, rates_te["GLM (tuned)"] * expo_te)
d_hgb = poisson_unit_dev(claims_te, rates_te["HGB (tuned)"] * expo_te)
point, lo, hi = paired_bootstrap(d_glm, d_hgb, expo_te, B=B)
log()
log("## GLM (tuned) contro HGB (tuned): differenza di devianza, IC bootstrap al 95% sulle polizze del test")
log()
log(f"Delta = devianza GLM - devianza HGB = {point:+.5f} per anno-polizza (IC: {lo:+.5f}, {hi:+.5f}). "
    "Negativo = GLM meglio; positivo = HGB meglio; se l'IC contiene 0 la differenza non e' distinguibile dal rumore.")

# ------------------------------------------------------------------ calibrazione e Lorenz
best_name = "HGB (tuned)" if res.set_index("model").loc["HGB (tuned)", "dev_test"] < \
    res.set_index("model").loc["GLM (tuned)", "dev_test"] else "GLM (tuned)"
log()
log(f"## Calibrazione per decili di esposizione sul test ({best_name}: tasso di sinistri osservato vs previsto)")
cal = decile_table(rates_te[best_name], claims_te, expo_te)
log()
block(cal)
cal.to_csv(REPORTS / f"m3_frequency_calibration{SUF}.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for name, style in [("GLM (tuned)", "o-"), ("HGB (tuned)", "s-")]:
    t = decile_table(rates_te[name], claims_te, expo_te)
    axes[0].plot(t["predicted"], t["observed"], style, label=name)
lim = [0, max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])]
axes[0].plot(lim, lim, "--", color="grey", lw=1)
axes[0].set_xlabel("frequenza prevista (media del decile)")
axes[0].set_ylabel("frequenza osservata")
axes[0].set_title("Calibrazione per decili (test)")
axes[0].legend()
for name in ["GLM (tuned)", "HGB (tuned)"]:
    g, ce, cl = lorenz_gini(rates_te[name], claims_te, expo_te)
    axes[1].plot(ce, cl, label=f"{name}: Gini {g:.3f}")
axes[1].plot([0, 1], [0, 1], "--", color="grey", lw=1)
axes[1].set_xlabel("esposizione cumulata (rischio crescente)")
axes[1].set_ylabel("sinistri cumulati")
axes[1].set_title("Curva di Lorenz (test)")
axes[1].legend()
fig.tight_layout()
fig.savefig(REPORTS / "figures" / f"m3_frequency_calibration_lorenz{SUF}.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ importanza delle variabili
sub = te.sample(min(60_000, len(te)), random_state=0)
pi = permutation_importance(
    fitted["HGB (tuned)"], frame(sub), sub[TARGET] / sub["exposure"],
    scoring="neg_mean_poisson_deviance", n_repeats=3, random_state=0, sample_weight=sub["exposure"],
)
imp = pd.DataFrame({"feature": frame(sub).columns, "dev_increase": pi.importances_mean,
                    "std": pi.importances_std}).sort_values("dev_increase", ascending=False)
log()
log("## Importanza per permutazione (HGB tuned, 60.000 polizze del test): aumento di devianza se si mescola la variabile")
log()
block(imp)
imp.to_csv(REPORTS / f"m3_frequency_importance{SUF}.csv", index=False)

# ------------------------------------------------------------------ salvataggi
joblib.dump(fitted["GLM (tuned)"], MODELS / f"frequency_glm{SUF}.joblib")
joblib.dump(fitted["HGB (tuned)"], MODELS / f"frequency_hgb{SUF}.joblib")
meta = {
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "sklearn_version": sklearn.__version__,
    "features": {"numeric": NUM, "categorical": CAT},
    "target": f"{TARGET} / exposure (tasso per anno-polizza), peso = exposure",
    "glm_params": gp,
    "hgb_params": hp,
    "baseline_rate_train": float(rate_null),
    "test_metrics": res.set_index("model")[["dev_test", "D2_test_%", "gini_test"]].round(5).to_dict("index"),
}
(MODELS / f"frequency_metadata{SUF}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

pred = te[["id_pol", "exposure", "claim_nb", "total_claim_amount"]].copy()
pred["rate_null"] = rates_te["Null"]
pred["rate_glm"] = rates_te["GLM (tuned)"]
pred["rate_hgb"] = rates_te["HGB (tuned)"]
pred.to_parquet(D / f"pred_freq_test{SUF}.parquet", index=False)

(REPORTS / f"m3_frequency{SUF}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nReport scritto in {REPORTS / 'm3_frequency.md'}")
