"""Milestone 7B: valore operativo del triage dei reclami (ore di revisione e casi gravi trovati).

Misurato (dal test, campione pesato come in 07_compare_cascade.py): recall dei casi gravi a un dato tasso di revisione per
TF-IDF (stadio 1), transformer (stadio 2) e combinazione, con IC bootstrap. Volumi annui dal 2025 (ultimo anno intero del test).
Assunzioni (assumptions.json): minuti per revisione, costo orario, costi di sviluppo e manutenzione.
Output: reports/m7_triage_value.md e reports/m7_triage_value.json. Gira nel container spark. Python 3.8 compatibile.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nlp"))
from ft_utils import logit, recall_at  # noqa: E402

NLP = Path(os.getenv("NLP_DIR", "data/nlp"))
REPORTS = Path(os.getenv("REPORTS_DIR", "reports"))
MODELS = Path(os.getenv("MODELS_DIR", "models"))
ASSUME = Path(os.getenv("ASSUMPTIONS", str(Path(__file__).resolve().parent / "assumptions.json")))
B = int(os.getenv("BOOTSTRAP_B", "200"))
YEAR = int(os.getenv("VOLUME_YEAR", "2025"))
REPORTS.mkdir(parents=True, exist_ok=True)
lines = []


def log(text=""):
    print(text, flush=True)
    lines.append(text)


T = json.loads(ASSUME.read_text(encoding="utf-8"))["triage"]
R = T["review_rate"]
nar = pd.read_parquet(NLP / "narratives", columns=["odino", "serious", "split", "year"])
pred = pd.read_parquet(NLP / "pred_stage2.parquet")
ref = json.loads((MODELS / "serving_reference.json").read_text(encoding="utf-8"))["nlp"]
bl = ref["blend"]

te_all = nar[nar["split"] == "test"]
N, P = len(te_all), int(te_all["serious"].sum())
yr = te_all[te_all["year"] == YEAR]
N_year, P_year = len(yr), int(yr["serious"].sum())
te = pred[pred["split"] == "test"].reset_index(drop=True)
te["score3"] = 1 / (1 + np.exp(-(bl["intercept"] + bl["coef_stage1"] * logit(te["score1"]) + bl["coef_stage2"] * logit(te["score2"]))))
y, w = te["serious"].to_numpy(), te["w"].to_numpy()
RANKERS = {"TF-IDF (stadio 1)": "score1", "Transformer (stadio 2)": "score2", "Combinazione (servizio)": "score3"}
RATES = [0.02, 0.05, R, 0.10]


def rec(idx, col, rate):
    return recall_at(y[idx], te[col].to_numpy()[idx], w[idx], N, P, rate)


all_idx = np.arange(len(te))
table = []
for name, col in RANKERS.items():
    for r in RATES:
        table.append({"ranker": name, "tasso_revisione": r, "recall": rec(all_idx, col, r)})
for r in RATES:
    table.append({"ranker": "Casuale (nessun modello)", "tasso_revisione": r, "recall": r})
tab = pd.DataFrame(table)

rng = np.random.default_rng(7)
boot = {k: [] for k in ("score1", "score3", "diff")}
for _ in range(B):
    idx = rng.integers(0, len(te), len(te))
    a, c = rec(idx, "score1", R), rec(idx, "score3", R)
    boot["score1"].append(a)
    boot["score3"].append(c)
    boot["diff"].append(c - a)
ci = {k: [float(v) for v in np.percentile(x, [2.5, 97.5])] for k, x in boot.items()}
r1, r3 = rec(all_idx, "score1", R), rec(all_idx, "score3", R)

log("# Milestone 7B: valore operativo del triage dei reclami")
log()
log(f"Test: {N:,} reclami e {P:,} esiti gravi. Volume annuo di riferimento ({YEAR}): {N_year:,} reclami, {P_year:,} esiti gravi ({P_year / N_year:.2%}). "
    f"Tasso di revisione del servizio: {R:.1%}. Bootstrap B={B}.")
log()
log("## 1. Recall dei casi gravi per tasso di revisione (misurato)")
log()
log("```text")
log(tab.pivot(index="ranker", columns="tasso_revisione", values="recall").round(3).to_string())
log("```")

reviewed = R * N_year
hours = reviewed * T["minutes_per_review"] / 60
hours_all = N_year * T["minutes_per_review"] / 60
cost_year = hours * T["reviewer_cost_per_hour"]
cost_all = hours_all * T["reviewer_cost_per_hour"]
found3, found1 = r3 * P_year, r1 * P_year
missed = (1 - r3) * P_year
extra = (r3 - r1) * P_year
extra_ci = [x * P_year for x in ci["diff"]]
maint = T["maintenance_per_year"]
build = T["build_cost"]

log()
log("## 2. Che cosa cambia in un anno (volumi del " + str(YEAR) + ")")
log()
rows = [
    ("Reclami riletti con il triage", f"{reviewed:,.0f} ({R:.1%})"),
    ("Ore di revisione con il triage / senza (rileggere tutto)", f"{hours:,.0f} / {hours_all:,.0f}  ({hours / T['hours_per_fte_year']:.1f} / {hours_all / T['hours_per_fte_year']:.1f} FTE)"),
    ("Costo di revisione con il triage / senza", f"{cost_year:,.0f} / {cost_all:,.0f} euro"),
    ("Risparmio rispetto a rileggere tutto", f"{cost_all - cost_year:,.0f} euro"),
    ("Casi gravi trovati (combinazione)", f"{found3:,.0f} su {P_year:,} (recall {r3:.1%}, IC {ci['score3'][0]:.1%}..{ci['score3'][1]:.1%})"),
    ("Casi gravi non trovati (rispetto a rileggere tutto)", f"{missed:,.0f}"),
    ("Casi gravi trovati rileggendo a caso lo stesso volume", f"{R * P_year:,.0f}"),
    ("Casi gravi in piu' del transformer rispetto al solo TF-IDF, stesso volume", f"{extra:,.0f} (IC {extra_ci[0]:,.0f}..{extra_ci[1]:,.0f})"),
]
for k, v in rows:
    log(f"- {k}: {v}")

log()
log("## 3. Punti di pareggio (non servono assunzioni sul valore di un caso)")
log()
log(f"- **Triage contro rileggere tutto**: conviene finche' un caso grave mancato costa meno di {(cost_all - cost_year) / max(missed, 1):,.0f} euro "
    f"({cost_all - cost_year:,.0f} euro risparmiati su {missed:,.0f} casi non trovati; al lordo di sviluppo e manutenzione, che considera 7C). "
    "Oltre quel valore conviene rileggere tutto.")
log(f"- **Transformer contro solo TF-IDF**: porta {extra:,.0f} casi in piu' l'anno (IC {extra_ci[0]:,.0f}..{extra_ci[1]:,.0f}) allo stesso volume di lavoro. "
    f"Il costo aggiuntivo e' la manutenzione ({maint:,.0f} euro l'anno) e lo sviluppo ({build:,.0f} euro, su 3 anni {build / 3:,.0f}): "
    f"conviene se un caso in piu' vale piu' di {(maint + build / 3) / max(extra, 1):,.0f} euro"
    + (f" (con il limite basso dell'IC: {(maint + build / 3) / max(extra_ci[0], 1e-9):,.0f} euro)." if extra_ci[0] > 0 else "; l'IC include lo zero: il vantaggio non e' dimostrato a questo volume."))
log()
log("Il valore di un caso grave trovato in tempo non e' un dato di questo progetto (dipende da regolatore, responsabilita' e riserve): i punti di pareggio dicono quanto deve valere perche' ogni scelta si ripaghi.")

out = {"year": YEAR, "N_year": N_year, "P_year": P_year, "review_rate": R, "recall_combo": r3, "recall_tfidf": r1,
       "recall_combo_ci": ci["score3"], "extra_cases": extra, "extra_cases_ci": extra_ci, "missed_vs_all": missed,
       "review_hours": hours, "review_hours_all": hours_all, "cost_year": cost_year, "cost_all": cost_all,
       "saving_vs_all": cost_all - cost_year, "table": table}
(REPORTS / "m7_triage_value.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
(REPORTS / "m7_triage_value.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nReport scritto in {REPORTS / 'm7_triage_value.md'}")
