"""Milestone 7A: valore economico della segmentazione e del passaggio da GLM a gradient boosting (selezione avversa).

Usa le polizze del TEST con i costi reali osservati (importi con cap per sinistro; i sinistri contati senza importo
valgono una quota `unpaid` del costo medio di un sinistro pagato). Simula un mercato a due assicuratori (bizlib.market):
ognuno ribasa il livello dei prezzi, quindi conta solo la differenza di relativita'. Coppie: HGB contro GLM, GLM contro
prezzo piatto (nessuna segmentazione), HGB contro prezzo piatto.

Per ogni coppia (H, G) e ogni cella (sinistri senza importo x mercato) riporta, in % del premio e con IC bootstrap:
  delta_h: variazione del risultato di H, che ha il modello migliore (valore OFFENSIVO: quasi zero, resta al margine obiettivo);
  delta_g: variazione del risultato di G, che ha il modello peggiore (costo di restare indietro = valore DIFENSIVO di H).
Output: reports/m7_pricing_value.md, reports/m7_pricing_value.json, reports/figures/m7_pricing_value.png.
Gira nel container spark (pandas, numpy, matplotlib). Python 3.8 compatibile.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bizlib  # noqa: E402

D = Path(os.getenv("MODELING_DIR", "data/modeling"))
REPORTS = Path(os.getenv("REPORTS_DIR", "reports"))
MODELS = Path(os.getenv("MODELS_DIR", "models"))
ASSUME = Path(os.getenv("ASSUMPTIONS", str(Path(__file__).resolve().parent / "assumptions.json")))
B = int(os.getenv("BOOTSTRAP_B", "200"))
SUF = "_payment"
(REPORTS / "figures").mkdir(parents=True, exist_ok=True)
lines = []


def log(text=""):
    print(text, flush=True)
    lines.append(text)


def block(df):
    log("```text")
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}", "display.width", 220):
        log(df.to_string(index=False))
    log("```")


A = json.loads(ASSUME.read_text(encoding="utf-8"))
fin = A["financial"]
LR, EXP = fin["loss_ratio_target"], fin["expense_ratio"]

# ------------------------------------------------------------------ dati
pm = json.loads((MODELS / f"pricing_metadata{SUF}.json").read_text(encoding="utf-8"))
cap, sev_const = float(pm["cap_value"]), float(pm["severity_constant_capped"])
pol = pd.read_parquet(D / "policies", columns=["id_pol", "split", "exposure", "claim_nb", "sev_claim_count"])
pred = pd.read_parquet(D / f"pred_freq_test{SUF}.parquet", columns=["id_pol", "rate_null", "rate_glm", "rate_hgb"])
te = pol[pol["split"] == "test"].merge(pred, on="id_pol", how="inner").reset_index(drop=True)
cl = pd.read_parquet(D / "claims", columns=["id_pol", "claim_amount", "split"])
cl = cl[cl["split"] == "test"]
capped = np.minimum(cl["claim_amount"].astype("float64"), cap).groupby(cl["id_pol"]).sum()
te["capped_loss"] = te["id_pol"].map(capped).fillna(0.0).to_numpy()
te["unpaid"] = np.maximum(te["claim_nb"].astype(float) - te["sev_claim_count"].astype(float), 0.0)
expo = te["exposure"].to_numpy(dtype=float)
unpaid_total, paid_total = float(te["unpaid"].sum()), float(te["sev_claim_count"].sum())

log("# Milestone 7A: valore economico del prezzo (segmentazione e passaggio da GLM a boosting)")
log()
log(f"Test: {len(te):,} polizze, {expo.sum():,.0f} anni-polizza, {paid_total:,.0f} sinistri pagati e {unpaid_total:,.0f} contati senza importo. "
    f"Costo per sinistro: importo reale con cap a {cap:,.0f}. Loss ratio obiettivo {LR:.0%}, spese {EXP:.0%}. Bootstrap B={B} sulle polizze.")
log("I guadagni sono in % del premio del caso base (stesso modello per entrambi): cio' che conta e' l'ordine di grandezza e se l'IC esclude lo zero. "
    "I costi sono quelli osservati nel test: con circa 5.000 sinistri il rumore e' grande e i risultati piccoli non sono distinguibili da zero.")

PAIRS = [("HGB contro GLM", "rate_hgb", "rate_glm"), ("GLM contro prezzo piatto", "rate_glm", "rate_null"),
         ("HGB contro prezzo piatto", "rate_hgb", "rate_null")]
cells, t0 = [], time.time()
for u in A["pricing"]["unpaid_cost_share"]:
    cost = te["capped_loss"].to_numpy() + u * te["unpaid"].to_numpy() * sev_const
    for mname, mk in A["pricing"]["markets"].items():
        for pname, h, g in PAIRS:
            r = bizlib.market_bootstrap(te[h].to_numpy(), te[g].to_numpy(), cost, expo, mk["shop"], mk["beta"], EXP, LR, B=B, seed=1)
            cells.append({"pair": pname, "unpaid": u, "market": mname, "shop": mk["shop"], "beta": mk["beta"], **r})
    print(f"[tempo] sinistri senza importo = {u}: {time.time() - t0:.0f}s", flush=True)
df = pd.DataFrame(cells)


def table(pair):
    d = df[df["pair"] == pair]
    return pd.DataFrame({
        "senza_importo": d["unpaid"], "mercato": d["market"],
        "perdente_%": d["delta_g_pct"], "perdente_ic": [f"{a:+.2f}..{b:+.2f}" for a, b in d["delta_g_ci"]],
        "migliore_%": d["delta_h_pct"], "migliore_ic": [f"{a:+.2f}..{b:+.2f}" for a, b in d["delta_h_ci"]],
        "quota_migliore": d["share_h"], "LR_perdente": d["loss_ratio_g"]})


for i, (pname, _, _) in enumerate(PAIRS, 1):
    log()
    log(f"## {i}. {pname}: effetto sul risultato tecnico (% del premio del caso base)")
    log()
    log("`perdente` = il modello peggiore (costo di restare indietro); `migliore` = il modello migliore (guadagno di volume e margine).")
    block(table(pname))

log()
log("## Lettura")
hg = df[(df["pair"] == "HGB contro GLM") & (df["market"] == "media") & (df["unpaid"] == 0.5)].iloc[0]
log(f"- HGB contro GLM, mercato medio e sinistri senza importo al 50%: chi resta con il GLM perde {-hg['delta_g_pct']:.2f}% del premio "
    f"(IC {hg['delta_g_ci'][0]:+.2f}..{hg['delta_g_ci'][1]:+.2f}); chi ha il boosting ottiene {hg['delta_h_pct']:+.2f}% "
    f"(IC {hg['delta_h_ci'][0]:+.2f}..{hg['delta_h_ci'][1]:+.2f}).")
fl = df[(df["pair"] == "GLM contro prezzo piatto") & (df["market"] == "media") & (df["unpaid"] == 0.5)].iloc[0]
ratio = fl["delta_g_pct"] / hg["delta_g_pct"] if hg["delta_g_pct"] else float("inf")
log(f"- Segmentare con un GLM contro non segmentare: chi non segmenta perde {-fl['delta_g_pct']:.2f}% del premio nello stesso scenario, "
    f"{ratio:.1f} volte il costo di restare col GLM invece del boosting"
    + (": la segmentazione di base vale molto piu' del passaggio da GLM a boosting." if ratio >= 2 else ": i due effetti hanno lo stesso ordine di grandezza."))
log("- Un modello preciso non guadagna molto oltre il margine obiettivo: il suo valore e' soprattutto difensivo, perche' evita le polizze sottoprezzate dal concorrente.")
log("- I sinistri senza importo sono una fonte di incertezza: se costano e sono concentrati in alcuni segmenti, un modello sul solo target dei sinistri pagati "
    "li sottoprezza (vedi la dipendenza dal parametro nelle tabelle).")

# ------------------------------------------------------------------ grafico
def cost_err(d):
    """Costo di restare indietro = -delta del perdente, con il suo IC; restituisce (valori, barre di errore asimmetriche)."""
    cost = -d["delta_g_pct"].to_numpy(dtype=float)
    lo = -np.array([c[1] for c in d["delta_g_ci"]])
    hi = -np.array([c[0] for c in d["delta_g_ci"]])
    return cost, [np.maximum(cost - lo, 0), np.maximum(hi - cost, 0)]


fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, (pname, _, _) in enumerate(PAIRS):
    d = df[(df["pair"] == pname) & (df["unpaid"] == 0.5)]
    cost, err = cost_err(d)
    ax[0].errorbar(np.arange(len(d)) + 0.12 * k, cost, yerr=err, fmt="o", capsize=3, label=pname)
ax[0].set_xticks(range(len(A["pricing"]["markets"])))
ax[0].set_xticklabels(list(A["pricing"]["markets"]))
ax[0].set_ylabel("costo di restare indietro (% del premio)")
ax[0].set_xlabel("sensibilita' del mercato al prezzo")
ax[0].set_title("Costo del modello peggiore (sinistri senza importo al 50%)")
ax[0].axhline(0, color="grey", lw=1)
ax[0].legend(fontsize=7)
d = df[(df["pair"] == "HGB contro GLM") & (df["market"] == "media")]
cost, err = cost_err(d)
ax[1].errorbar(d["unpaid"], cost, yerr=err, fmt="o-", capsize=3)
ax[1].set_xlabel("costo dei sinistri senza importo (quota del sinistro pagato)")
ax[1].set_ylabel("costo di restare col GLM (% del premio)")
ax[1].set_title("HGB contro GLM, mercato medio")
ax[1].axhline(0, color="grey", lw=1)
fig.tight_layout()
fig.savefig(REPORTS / "figures" / "m7_pricing_value.png", dpi=130)
plt.close(fig)

out = {"cap": cap, "severity_const": sev_const, "exposure_test": float(expo.sum()), "unpaid_total": unpaid_total, "paid_total": paid_total,
       "cells": [{k: (v if not isinstance(v, np.floating) else float(v)) for k, v in c.items()} for c in cells]}
(REPORTS / "m7_pricing_value.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
(REPORTS / "m7_pricing_value.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nReport scritto in {REPORTS / 'm7_pricing_value.md'}")
