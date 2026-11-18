"""Milestone 5A - Audit dei rischi dei modelli assicurativi (frequenza dei sinistri pagati, GLM e gradient boosting).

1. Calibrazione ed equita' per gruppo (eta' del conducente, area, regione, eta' del veicolo, Bonus-Malus, marca, alimentazione)
2. Interruzione di una variabile (fallimento operativo): cosa succede se in produzione arriva vuota o sbagliata
3. Overfitting: curva di apprendimento e curva per iterazione del gradient boosting
4. Spostamenti del portafoglio: PSI e errore di premio sotto composizioni diverse (calibra le soglie di allarme)
5. Rilevabilita': quanti sinistri servono per accorgersi di una deriva della frequenza

Tutto sul test (mai usato per scegliere i modelli). Richiede 03/01_frequency.py con FREQ_TARGET=sev_claim_count
(file con suffisso _payment). Output: reports/m5_insurance_audit.md, reports/figures/m5_*.png, reports/m5_group_calibration.csv.
"""
import json
import os
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modeling"))
from mlib import (  # noqa: E402
    CAT,
    NUM,
    dev_per_exposure,
    fit_weighted,
    frame,
    lorenz_gini,
    make_glm,
    make_hgb,
    poisson_unit_dev,
    to_text,
)
from risk_lib import (  # noqa: E402
    AGE_BINS,
    AGE_LABELS,
    BM_BINS,
    BM_LABELS,
    VEH_AGE_BINS,
    VEH_AGE_LABELS,
    band,
    claims_needed,
    flag_outside,
    group_calibration,
    psi,
    psi_categorical,
)

D = Path(os.getenv("MODELING_DIR", "data/modeling"))
REPORTS = Path(os.getenv("REPORTS_DIR", "reports"))
MODELS = Path(os.getenv("MODELS_DIR", "models"))
B = int(os.getenv("BOOTSTRAP_B", "200"))
SUF = "_payment"
(REPORTS / "figures").mkdir(parents=True, exist_ok=True)
lines = []


def log(text=""):
    print(text, flush=True)
    lines.append(text)


def block(df):
    log("```text")
    log(to_text(df))
    log("```")


# ------------------------------------------------------------------ dati e modelli
pol = pd.read_parquet(D / "policies")
for c in CAT + ["area"]:
    pol[c] = pol[c].astype(str)
for c in ["exposure", "claim_nb", "sev_claim_count"] + NUM:
    pol[c] = pol[c].astype("float64")
pred = pd.read_parquet(D / f"pred_freq_test{SUF}.parquet", columns=["id_pol", "rate_glm", "rate_hgb"])
tr = pol[pol["split"] == "train"].reset_index(drop=True)
te = pol[pol["split"] == "test"].merge(pred, on="id_pol", how="inner").reset_index(drop=True)
meta = json.loads((MODELS / f"frequency_metadata{SUF}.json").read_text(encoding="utf-8"))
glm = joblib.load(MODELS / f"frequency_glm{SUF}.joblib")
hgb = joblib.load(MODELS / f"frequency_hgb{SUF}.joblib")

obs, expo = te["sev_claim_count"].to_numpy(), te["exposure"].to_numpy()
exp_glm, exp_hgb = te["rate_glm"].to_numpy() * expo, te["rate_hgb"].to_numpy() * expo
chk = max(float(np.abs(glm.predict(frame(te)) - te["rate_glm"]).max()), float(np.abs(hgb.predict(frame(te)) - te["rate_hgb"]).max()))

log("# Milestone 5A: audit dei rischi dei modelli assicurativi")
log()
log(f"Modelli: GLM e gradient boosting sul tasso di sinistri con importo (sinistri pagati). Test: {len(te):,} polizze, "
    f"{expo.sum():,.0f} anni-polizza, {obs.sum():,.0f} sinistri pagati (frequenza {obs.sum() / expo.sum():.4f}). "
    f"Bootstrap B={B}. Coerenza tra previsioni salvate e modelli ricaricati: differenza massima {chk:.2e}.")
if chk > 1e-6:
    log("ATTENZIONE: le previsioni salvate non coincidono con quelle dei modelli ricaricati (versioni diverse delle librerie?).")

# ------------------------------------------------------------------ 1. calibrazione ed equita' per gruppo
te["eta_conducente"] = band(te["driv_age_c"], AGE_BINS, AGE_LABELS)
te["eta_veicolo"] = band(te["veh_age_c"], VEH_AGE_BINS, VEH_AGE_LABELS)
te["bonus_malus"] = band(te["bonus_malus_c"], BM_BINS, BM_LABELS)
top = te.groupby("region")["exposure"].sum().nlargest(8).index
te["regione"] = np.where(te["region"].isin(top), te["region"], "altre")
te["marca"] = np.where(te["veh_brand"] == "B12", "B12", "altre")
te["alimentazione"] = te["veh_gas"]
GROUPS = ["eta_conducente", "area", "regione", "eta_veicolo", "bonus_malus", "marca", "alimentazione"]
d_glm, d_hgb = poisson_unit_dev(obs, exp_glm), poisson_unit_dev(obs, exp_hgb)

tabs = []
for gcol in GROUPS:
    t = group_calibration(te[gcol], obs, {"GLM": exp_glm, "HGB": exp_hgb}, expo, B=B)
    codes, levels = pd.factorize(te[gcol].to_numpy(), sort=True)
    t["dev_GLM"] = np.bincount(codes, weights=d_glm) / t["anni_polizza"]
    t["dev_HGB"] = np.bincount(codes, weights=d_hgb) / t["anni_polizza"]
    t.insert(0, "variabile", gcol)
    t["sovrapprezzo_HGB_%"] = 100 * (1 / t["rapporto_HGB"] - 1)
    t["fuori_IC_HGB"] = flag_outside(t, "HGB")
    t["fuori_IC_GLM"] = flag_outside(t, "GLM")
    tabs.append(t)
allg = pd.concat(tabs, ignore_index=True)
allg.to_csv(REPORTS / "m5_group_calibration.csv", index=False)

log()
log("## 1. Calibrazione ed equita' per gruppo (test)")
log()
log("`rapporto` = sinistri osservati / sinistri attesi dal modello: 1 = calibrato, < 1 = il modello sovrastima il rischio del "
    "gruppo (il gruppo paga piu' del dovuto), > 1 = lo sottostima (e' sussidiato dagli altri). `sovrapprezzo` = previsto/osservato - 1. "
    "'*' = intervallo di confidenza al 95% che esclude 1. Con molti gruppi alcuni '*' sono attesi per caso (circa 1 su 20).")
for gcol, t in zip(GROUPS, tabs):
    log()
    log(f"Per {gcol}:")
    show = t[["gruppo", "anni_polizza", "osservati", "freq_osservata", "freq_prevista_HGB", "rapporto_HGB", "ic_low_HGB", "ic_high_HGB",
              "fuori_IC_HGB", "rapporto_GLM", "fuori_IC_GLM", "sovrapprezzo_HGB_%", "dev_GLM", "dev_HGB"]]
    block(show)
n_g, n_h, n_l = len(allg), int((allg["fuori_IC_HGB"] == "*").sum()), int((allg["fuori_IC_GLM"] == "*").sum())
log()
log(f"Gruppi con rapporto fuori dall'IC: HGB {n_h} su {n_g}, GLM {n_l} su {n_g} (per caso se ne attenderebbero circa {0.05 * n_g:.1f}).")
worst = allg.loc[(allg["rapporto_HGB"] - 1).abs().nlargest(5).index, ["variabile", "gruppo", "anni_polizza", "rapporto_HGB", "fuori_IC_HGB"]]
log("I cinque gruppi con la calibrazione peggiore (HGB):")
block(worst)

fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
for ax, gcol, t in zip(axes.ravel(), GROUPS, tabs):
    y = np.arange(len(t))
    for off, name, color in ((-0.15, "GLM", "tab:blue"), (0.15, "HGB", "tab:orange")):
        err = [t[f"rapporto_{name}"] - t[f"ic_low_{name}"], t[f"ic_high_{name}"] - t[f"rapporto_{name}"]]
        ax.errorbar(t[f"rapporto_{name}"], y + off, xerr=err, fmt="o", ms=4, capsize=2, color=color, label=name)
    ax.axvline(1.0, color="grey", ls="--", lw=1)
    ax.set_yticks(y, t["gruppo"])
    ax.set_title(gcol, fontsize=10)
axes.ravel()[0].legend(fontsize=8)
for ax in axes.ravel()[len(GROUPS):]:
    ax.axis("off")
fig.suptitle("Osservati / attesi per gruppo (1 = calibrato), IC bootstrap 95%")
fig.tight_layout()
fig.savefig(REPORTS / "figures" / "m5_group_calibration.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ 2. interruzione di una variabile
log()
log("## 2. Interruzione di una variabile: cosa succede se in produzione arriva vuota o sbagliata")
log()
log("Ogni variabile viene sostituita, da sola, con la mediana del train (numeriche) o con un livello sconosciuto (categoriche). "
    "`degrado_%` = aumento della devianza di Poisson rispetto ai dati integri; `rapporto` = osservati/attesi sull'intero portafoglio.")
med = {c: float(tr[c].median()) for c in NUM}
rows = []
for name, model in (("GLM", glm), ("HGB", hgb)):
    base_rate = model.predict(frame(te))
    base_dev = dev_per_exposure(obs, base_rate, expo)
    rows.append({"modello": name, "variabile_interrotta": "(nessuna)", "devianza": base_dev, "degrado_%": 0.0,
                 "gini": lorenz_gini(base_rate, obs, expo)[0], "rapporto": obs.sum() / (base_rate * expo).sum()})
    for col in NUM + CAT:
        broken = te.copy()
        broken[col] = med[col] if col in NUM else "UNKNOWN"
        rate = model.predict(frame(broken))
        dev = dev_per_exposure(obs, rate, expo)
        rows.append({"modello": name, "variabile_interrotta": col, "devianza": dev, "degrado_%": 100 * (dev - base_dev) / base_dev,
                     "gini": lorenz_gini(rate, obs, expo)[0], "rapporto": obs.sum() / (rate * expo).sum()})
out = pd.DataFrame(rows)
block(out)
worst_h = out[(out["modello"] == "HGB") & (out["variabile_interrotta"] != "(nessuna)")].nlargest(3, "degrado_%")
log(f"Le variabili la cui interruzione costa di piu' (HGB): {', '.join(worst_h['variabile_interrotta'])}. "
    "Per queste serve un controllo di qualita' dei dati in ingresso, con allarme se il valore manca o e' costante.")
out.to_csv(REPORTS / "m5_feature_outage.csv", index=False)

# ------------------------------------------------------------------ 3. overfitting
log()
log("## 3. Overfitting: curva di apprendimento e curva per iterazione")
fracs = [0.02, 0.05, 0.10, 0.25, 0.50, 1.0]
perm = np.random.default_rng(0).permutation(len(tr))
rows = []
for f in fracs:
    sub = tr.iloc[perm[: max(1000, int(f * len(tr)))]]
    y_sub, w_sub = (sub["sev_claim_count"] / sub["exposure"]).to_numpy(), sub["exposure"].to_numpy()
    for name, model in (("GLM", make_glm("poisson", **meta["glm_params"])), ("HGB", make_hgb("poisson", **meta["hgb_params"]))):
        t0 = time.time()
        fit_weighted(model, frame(sub), y_sub, w_sub)
        d_tr = dev_per_exposure(sub["sev_claim_count"].to_numpy(), model.predict(frame(sub)), w_sub)
        d_te = dev_per_exposure(obs, model.predict(frame(te)), expo)
        rows.append({"modello": name, "quota_train_%": 100 * f, "polizze": len(sub), "dev_train": d_tr, "dev_test": d_te,
                     "scarto_%": 100 * (d_te - d_tr) / d_tr})
        print(f"[tempo] curva {name} {f:.0%}: {time.time() - t0:.0f}s", flush=True)
lc = pd.DataFrame(rows)
log()
log("Devianza di Poisson per anno-polizza, addestrando su una frazione del train (parametri fissi, quelli scelti con Optuna):")
block(lc)
log("Uno scarto che si restringe al crescere dei dati e un test che continua a migliorare indicano che altri dati aiuterebbero; "
    "un test che si appiattisce con uno scarto che resta largo indica che serve piu' regolarizzazione, non piu' dati.")

prep, mdl = hgb.named_steps["prep"], hgb.named_steps["model"]
samp = tr.sample(min(100_000, len(tr)), random_state=0)
Xs, Xt = prep.transform(frame(samp)), prep.transform(frame(te))
y_s, e_s = samp["sev_claim_count"].to_numpy(), samp["exposure"].to_numpy()
dev_s, dev_t = [], []
for ps, pt in zip(mdl.staged_predict(Xs), mdl.staged_predict(Xt)):
    dev_s.append(dev_per_exposure(y_s, ps, e_s))
    dev_t.append(dev_per_exposure(obs, pt, expo))
dev_s, dev_t = np.array(dev_s), np.array(dev_t)
best_it = int(dev_t.argmin()) + 1
log()
log(f"Gradient boosting: {len(dev_t)} iterazioni selezionate dall'early stopping interno; la devianza di test e' minima all'iterazione "
    f"{best_it} ({dev_t.min():.5f}) e alla fine vale {dev_t[-1]:.5f} (differenza {100 * (dev_t[-1] / dev_t.min() - 1):+.2f}%). "
    f"Scarto finale train-test: {100 * (dev_t[-1] - dev_s[-1]) / dev_s[-1]:.1f}%.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, c in (("GLM", "tab:blue"), ("HGB", "tab:orange")):
    s = lc[lc["modello"] == name]
    axes[0].plot(s["polizze"], s["dev_train"], "--", color=c, label=f"{name} train")
    axes[0].plot(s["polizze"], s["dev_test"], "-o", color=c, label=f"{name} test")
axes[0].set_xscale("log")
axes[0].set_xlabel("polizze di addestramento")
axes[0].set_ylabel("devianza di Poisson per anno-polizza")
axes[0].set_title("Curva di apprendimento")
axes[0].legend(fontsize=8)
axes[1].plot(np.arange(1, len(dev_s) + 1), dev_s, "--", label="train (campione)")
axes[1].plot(np.arange(1, len(dev_t) + 1), dev_t, label="test")
axes[1].set_xlabel("iterazioni di boosting")
axes[1].set_title("HGB: devianza per iterazione")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(REPORTS / "figures" / "m5_learning_curve.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ 4. spostamenti del portafoglio e PSI
log()
log("## 4. Spostamenti del portafoglio: PSI e errore di premio (per calibrare le soglie di allarme)")
rate_tr = hgb.predict(frame(tr))
base_psi = {"punteggio (frequenza prevista)": psi(rate_tr, te["rate_hgb"]), "eta_conducente": psi(tr["driv_age_c"], te["driv_age_c"]),
            "eta_veicolo": psi(tr["veh_age_c"], te["veh_age_c"]), "bonus_malus": psi(tr["bonus_malus_c"], te["bonus_malus_c"]),
            "area": psi_categorical(tr["area"], te["area"])}
log()
log("Riferimento: PSI tra train e test (lo split e' casuale, quindi ci si attende un valore vicino a zero; e' il pavimento del rumore):")
block(pd.DataFrame({"variabile": list(base_psi), "psi": list(base_psi.values())}))

scen = [("Piu' giovani (18-24)", te["driv_age_c"] < 25, "driv_age_c"), ("Auto piu' vecchie (11+ anni)", te["veh_age_c"] >= 11, "veh_age_c"),
        ("Bonus-malus peggiore (>75)", te["bonus_malus_c"] > 75, "bonus_malus_c")]
rows = []
for name, mask, feat in scen:
    for k in (1.0, 1.5, 2.0, 3.0, 5.0):
        w = np.where(mask, k, 1.0)
        rows.append({
            "scenario": name, "moltiplicatore": k, "quota_gruppo_%": 100 * w[mask].sum() / w.sum(),
            "psi_punteggio": psi(rate_tr, te["rate_hgb"], new_weights=w), "psi_variabile": psi(tr[feat], te[feat], new_weights=w),
            "var_freq_prevista_%": 100 * ((w * exp_hgb).sum() / (w * expo).sum() / (exp_hgb.sum() / expo.sum()) - 1),
            "rapporto_oss_prev": (w * obs).sum() / (w * exp_hgb).sum()})
sc = pd.DataFrame(rows)
log()
log("Si ripesa il test per simulare una composizione diversa del portafoglio (shift delle covariate): il modello resta calibrato "
    "per costruzione se le relazioni non cambiano, ma il livello medio del premio si sposta. Il PSI sul punteggio dice a quale "
    "spostamento corrisponde una certa soglia di allarme:")
block(sc)
sc.to_csv(REPORTS / "m5_shift_psi.csv", index=False)
log("Lettura: con PSI < 0,10 il portafoglio e' stabile; le soglie 0,10 e 0,25 sono convenzioni, e la tabella mostra a quale variazione "
    "della frequenza prevista e della composizione corrispondono in questo portafoglio.")

fig, ax = plt.subplots(figsize=(6.4, 4.2))
for name, _, _ in scen:
    s = sc[sc["scenario"] == name]
    ax.plot(s["psi_punteggio"], s["var_freq_prevista_%"], "o-", label=name)
ax.axvline(0.10, color="grey", ls="--", lw=1)
ax.axvline(0.25, color="red", ls="--", lw=1)
ax.set_xlabel("PSI del punteggio")
ax.set_ylabel("variazione della frequenza prevista (%)")
ax.set_title("Soglie di PSI e variazione del premio")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(REPORTS / "figures" / "m5_psi_calibration.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ 5. rilevabilita'
log()
log("## 5. Rilevabilita': quanti sinistri servono per accorgersi di una deriva della frequenza")
log()
log("Approssimazione di Poisson: per rilevare uno spostamento relativo d (95% di confidenza, 80% di potenza) servono circa (2,8/d)^2 "
    "sinistri nel gruppo osservato. Gli anni-polizza si ricavano dalla frequenza del gruppo. E' un ordine di grandezza, non una garanzia.")
segs = {"intero portafoglio": np.ones(len(te), dtype=bool), "18-24 anni": (te["driv_age_c"] < 25).to_numpy(),
        "veicoli di eta' 0": (te["veh_age_c"] == 0).to_numpy(), "Bonus-malus > 100": (te["bonus_malus_c"] > 100).to_numpy()}
rows = []
for delta in (0.05, 0.10, 0.15, 0.20, 0.30):
    n = claims_needed(delta)
    for s, m in segs.items():
        rate = obs[m].sum() / expo[m].sum()
        rows.append({"deriva": f"{delta:.0%}", "gruppo": s, "freq_osservata": rate, "sinistri_necessari": n,
                     "anni_polizza_necessari": n / rate, "volte_esposizione_gruppo_nel_test": (n / rate) / expo[m].sum()})
block(pd.DataFrame(rows))
log("`volte_esposizione_gruppo_nel_test` = anni-polizza necessari divisi per l'esposizione che il gruppo ha nel test (> 1: servirebbe "
    "piu' esposizione di quella disponibile). Nei gruppi piccoli (veicoli nuovi, Bonus-malus alto) il monitoraggio per soglie e' lento: "
    "conviene accorpare periodi o usare test sequenziali.")

(REPORTS / "m5_insurance_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nReport scritto in {REPORTS / 'm5_insurance_audit.md'}")
