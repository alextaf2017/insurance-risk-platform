"""Milestone 7C: business case. Combina i risultati misurati (7A prezzo, 7B triage) con le assunzioni di scenario.

Per il passaggio da GLM a gradient boosting: tre scenari (pessimistico, base, ottimistico), VAN a 3 anni, rientro, punto di pareggio
del portafoglio e sensibilita' a una assunzione alla volta (diagramma a tornado). Per il triage: risparmio e punti di pareggio.
Output: reports/m7_business_case.md, m7_business_case_scenarios.csv, m7_business_case.json, figures/m7_business_case_tornado.png.
Richiede reports/m7_pricing_value.json e m7_triage_value.json. Python 3.8 compatibile.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bizlib  # noqa: E402

REPORTS = Path(os.getenv("REPORTS_DIR", "reports"))
ASSUME = Path(os.getenv("ASSUMPTIONS", str(Path(__file__).resolve().parent / "assumptions.json")))
(REPORTS / "figures").mkdir(parents=True, exist_ok=True)
lines = []


def log(text=""):
    print(text, flush=True)
    lines.append(text)


A = json.loads(ASSUME.read_text(encoding="utf-8"))
FIN, COSTS, SC, PR = A["financial"], A["costs"], A["scenarios"], A["pricing"]
H, RATE = int(FIN["horizon_years"]), FIN["discount_rate"]
PV = json.loads((REPORTS / "m7_pricing_value.json").read_text(encoding="utf-8"))
TV = json.loads((REPORTS / "m7_triage_value.json").read_text(encoding="utf-8"))
PAIR = "HGB contro GLM"


def cell(unpaid, market):
    cs = [c for c in PV["cells"] if c["pair"] == PAIR and c["market"] == market]
    return min(cs, key=lambda c: abs(c["unpaid"] - unpaid))


def evaluate(sc, book=None, bound=None):
    """VAN e indicatori di uno scenario. bound: None (stima puntuale), 'low' o 'high' (estremi dell'IC bootstrap del test)."""
    c = cell(sc["unpaid"], sc["market"])
    prem = c["premium_per_py"]
    dg, dh = c["delta_g_pct"], c["delta_h_pct"]
    if bound == "low":
        dg, dh = c["delta_g_ci"][1], c["delta_h_ci"][0]   # perdita del peggiore piu' piccola, guadagno del migliore piu' piccolo
    elif bound == "high":
        dg, dh = c["delta_g_ci"][0], c["delta_h_ci"][1]
    p = sc["p_competitors_upgrade"]
    per_py = (p * (-dg) + (1 - p) * dh) / 100 * prem               # euro per anno-polizza del mercato simulato (costi con cap)
    tail = PR["tail_load"][sc["tail"]]
    book = book or FIN["book_policy_years"]
    ben = bizlib.benefit_schedule(per_py, book, tail, sc["shrink"], sc["decay"], H)
    cost = bizlib.cost_schedule(COSTS, sc["cost_mult"], H)
    flows = [b + k for b, k in zip(ben, cost)]
    unit = [b / book for b in ben]
    return {"npv": bizlib.npv(flows, RATE), "payback": bizlib.payback_year(flows), "benefit_year1": ben[1], "cost_year0": -cost[0],
            "cost_run": -cost[1], "break_even_book": bizlib.break_even_book([-k for k in cost], unit, RATE),
            "per_py_market": per_py, "defensive_pct": -dg, "offensive_pct": dh, "flows": flows}


log("# Milestone 7: business case")
log()
log(f"Portafoglio illustrativo: {FIN['book_policy_years']:,} anni-polizza | VAN a {H} anni, tasso {RATE:.0%} | spese {FIN['expense_ratio']:.0%}, "
    f"loss ratio obiettivo {FIN['loss_ratio_target']:.0%}. **I valori di scenario (mercato, costi, quota di guadagno conservata) sono assunzioni, non dati**: "
    "stanno in `src/business/assumptions.json` e vanno sostituiti con quelli dell'azienda. Sono misurati: il costo di essere meno preciso del mercato (7A) e il comportamento del triage (7B).")

# ------------------------------------------------------------------ A. prezzo
log()
log("## A. Passare da GLM a gradient boosting per il prezzo")
rows = []
for name in ("pessimistico", "base", "ottimistico"):
    r = evaluate(SC[name])
    rows.append({"scenario": name, "costo_dev_anno0": r["cost_year0"], "costo_annuo": r["cost_run"], "beneficio_anno1": r["benefit_year1"],
                 "VAN": r["npv"], "rientro_anno": r["payback"], "pareggio_anni_polizza": r["break_even_book"]})
tab = pd.DataFrame(rows)
tab.to_csv(REPORTS / "m7_business_case_scenarios.csv", index=False)
log()
log("```text")
with pd.option_context("display.float_format", lambda v: f"{v:,.0f}", "display.width", 200):
    log(tab.to_string(index=False))
log("```")
base = evaluate(SC["base"])
lo, hi = evaluate(SC["base"], bound="low"), evaluate(SC["base"], bound="high")
log()
log(f"Nello scenario base il valore atteso viene per {100 * SC['base']['p_competitors_upgrade']:.0f}% dal costo evitato di restare col GLM "
    f"mentre i concorrenti migliorano (stima {base['defensive_pct']:.2f}% del premio) e per il resto dal guadagno di volume ({base['offensive_pct']:+.2f}%). "
    f"Estremi dell'IC bootstrap del test per lo scenario base: VAN da {lo['npv']:,.0f} a {hi['npv']:,.0f} euro.")
log("Attenzione: con circa 5.000 sinistri nel test il guadagno HGB contro GLM e' piccolo rispetto al rumore: se l'IC e' largo, e' il segnale piu' importante di questo business case.")

# beneficio minimo per il pareggio (non dipende dall'effetto misurato)
prem_base = cell(SC["base"]["unpaid"], SC["base"]["market"])["premium_per_py"]
unit_flows = bizlib.benefit_schedule(prem_base / 100, FIN["book_policy_years"], PR["tail_load"][SC["base"]["tail"]],
                                     SC["base"]["shrink"], SC["base"]["decay"], H)
pv_cost = -bizlib.npv(bizlib.cost_schedule(COSTS, SC["base"]["cost_mult"], H), RATE)
pv_unit = bizlib.npv(unit_flows, RATE)
req_pct = pv_cost / pv_unit
eff_base = SC["base"]["p_competitors_upgrade"] * base["defensive_pct"] + (1 - SC["base"]["p_competitors_upgrade"]) * base["offensive_pct"]
log()
log(f"**Beneficio minimo per il pareggio** (scenario base): serve un effetto medio di {req_pct:.2f}% del premio (mercato simulato) per VAN zero; "
    f"la stima del test, pesata sulle probabilita' di scenario, e' {eff_base:.2f}%. Il minimo dipende da costi, dimensione del portafoglio e premio medio, non dall'effetto misurato: "
    "se i due valori sono vicini, decidono le assunzioni.")

# sensibilita' a una assunzione alla volta (scenario base)
sens = []


def vary(label, mutate, values):
    for v in values:
        sc = copy.deepcopy(SC["base"])
        book = None
        book = mutate(sc, v) or book
        sens.append({"parametro": label, "valore": str(v), "VAN": evaluate(sc, book)["npv"]})


vary("sinistri senza importo (costo)", lambda sc, v: sc.update(unpaid=v), PR["unpaid_cost_share"])
vary("mercato (sensibilita' al prezzo)", lambda sc, v: sc.update(market=v), list(PR["markets"]))
vary("fattore di coda", lambda sc, v: sc.update(tail=v), list(PR["tail_load"]))
vary("guadagno conservato in produzione", lambda sc, v: sc.update(shrink=v), [0.4, 0.7, 1.0])
vary("concorrenti migliorano (probabilita')", lambda sc, v: sc.update(p_competitors_upgrade=v), [0.0, 0.5, 1.0])
vary("durata del vantaggio", lambda sc, v: sc.update(decay=v), [[1, 0.5, 0.25], [1, 0.75, 0.5], [1, 1, 1]])
vary("moltiplicatore dei costi", lambda sc, v: sc.update(cost_mult=v), [0.8, 1.0, 1.5])
vary("dimensione del portafoglio (anni-polizza)", lambda sc, v: v, [100_000, 300_000, 1_000_000])
sdf = pd.DataFrame(sens)
log()
log("### Sensibilita' del VAN (euro) a una assunzione alla volta, resto sul base")
log()
log("```text")
with pd.option_context("display.float_format", lambda v: f"{v:,.0f}", "display.width", 200):
    log(sdf.to_string(index=False))
log("```")
span = sdf.groupby("parametro")["VAN"].agg(["min", "max"])
SCALE = "dimensione del portafoglio (anni-polizza)"
span["ampiezza"] = span["max"] - span["min"]
span = span.sort_values("ampiezza")
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.barh(span.index, span["max"] - span["min"], left=span["min"], color="#4C78A8")
ax.axvline(base["npv"], color="black", lw=1)
ax.axvline(0, color="red", ls="--", lw=1)
ax.set_xlabel("VAN a 3 anni (euro): linea nera = base, rossa = pareggio")
ax.set_title("Quali assunzioni muovono il risultato")
fig.tight_layout()
fig.savefig(REPORTS / "figures" / "m7_business_case_tornado.png", dpi=130)
plt.close(fig)
top = span.drop(SCALE).index[-1]  # la dimensione del portafoglio e' una scala, non un'incertezza
log()
log(f"L'assunzione che sposta di piu' il VAN e' **{top}** (da {span.loc[top, 'min']:,.0f} a {span.loc[top, 'max']:,.0f} euro). "
    f"Punto di pareggio nello scenario base: {base['break_even_book']:,.0f} anni-polizza" if base["break_even_book"] else
    "Nello scenario base il beneficio non e' positivo: nessun portafoglio ripaga il costo.")

# ------------------------------------------------------------------ B. triage
T = A["triage"]
net_year = TV["saving_vs_all"] - T["maintenance_per_year"]
tflows = [-T["build_cost"]] + [net_year] * H
log()
log("## B. Triage dei reclami")
log()
log(f"Con volumi {TV['year']} ({TV['N_year']:,} reclami, {TV['P_year']:,} gravi): il triage rilegge il {TV['review_rate']:.1%} dei reclami e trova il "
    f"{TV['recall_combo']:.1%} dei casi gravi (IC {TV['recall_combo_ci'][0]:.1%}..{TV['recall_combo_ci'][1]:.1%}). "
    f"Rispetto a rileggere tutto risparmia {TV['saving_vs_all']:,.0f} euro l'anno ({TV['review_hours_all'] - TV['review_hours']:,.0f} ore) "
    f"e lascia {TV['missed_vs_all']:,.0f} casi gravi non riletti.")
log(f"VAN a {H} anni del triage contro rileggere tutto (costo di sviluppo {T['build_cost']:,.0f}, manutenzione {T['maintenance_per_year']:,.0f} l'anno): "
    f"{bizlib.npv(tflows, RATE):,.0f} euro, **prima di attribuire un costo ai casi mancati**. "
    f"Il triage e' preferibile a rileggere tutto finche' un caso grave mancato vale meno di "
    f"{(TV['saving_vs_all'] - T['maintenance_per_year'] - T['build_cost'] / H) / max(TV['missed_vs_all'], 1):,.0f} euro "
    "(risparmio annuo al netto di manutenzione e sviluppo ammortizzato, diviso i casi non riletti).")
log(f"Il transformer in piu' rispetto al solo TF-IDF trova {TV['extra_cases']:,.0f} casi in piu' l'anno "
    f"(IC {TV['extra_cases_ci'][0]:,.0f}..{TV['extra_cases_ci'][1]:,.0f}): vedi `reports/m7_triage_value.md` per il suo punto di pareggio.")

# ------------------------------------------------------------------ C. decisione
log()
log("## C. Che cosa dice il business case")
pos = [n for n in ("pessimistico", "base", "ottimistico") if evaluate(SC[n])["npv"] > 0]
log(f"- Prezzo, GLM contro boosting: VAN positivo in {len(pos)} scenari su 3 ({', '.join(pos) or 'nessuno'}).")
fl = [c for c in PV["cells"] if c["pair"] == "GLM contro prezzo piatto" and c["market"] == SC["base"]["market"]]
fl = min(fl, key=lambda c: abs(c["unpaid"] - SC["base"]["unpaid"]))
hg = cell(SC["base"]["unpaid"], SC["base"]["market"])
log(f"- Costo di non segmentare ({-fl['delta_g_pct']:.2f}% del premio) contro costo di restare col GLM invece del boosting ({-hg['delta_g_pct']:.2f}%), "
    "scenario base: " + ("la segmentazione di base vale molto piu' del passaggio al boosting." if fl["delta_g_pct"] / hg["delta_g_pct"] >= 2
                         else "i due effetti hanno lo stesso ordine di grandezza."))
log("- Prima di impegnare risorse: confermare con i titolari dei dati il costo dei sinistri senza importo, la coda e la quota di clienti che confrontano i prezzi.")
log("- Il costo di governance (validazione, monitoraggio, riaddestramento) pesa quanto lo sviluppo: il beneficio deve superarlo ogni anno, non solo una volta.")

out = {"scenarios": rows, "base": {k: v for k, v in base.items() if k != "flows"}, "base_ci_npv": [lo["npv"], hi["npv"]],
       "sensitivity": sens, "triage_npv": bizlib.npv(tflows, RATE), "triage_net_year": net_year}
(REPORTS / "m7_business_case.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
(REPORTS / "m7_business_case.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nReport scritto in {REPORTS / 'm7_business_case.md'}")
