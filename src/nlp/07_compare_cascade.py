"""Milestone 4, passo 6: confronto tra TF-IDF, transformer e loro combinazione (cascata a due stadi).

Tutte le metriche usano il campione pesato del test (tutti i positivi tra i candidati, una frazione dei negativi con
peso 1/frazione): sono stime della popolazione intera, con intervalli bootstrap sui reclami. Il recall al tasso di
revisione r e' calcolato sul totale dei reclami del test e sul totale dei positivi (anche quelli fuori dai candidati,
che la cascata non rilegge mai).

Output: reports/m4_cascade_comparison.md, reports/figures/m4_cascade_recall_curves.png,
reports/m4_cascade_recall.csv, data/nlp/error_examples.md (esempi di testo: NON committare, contengono dati personali).
Gira nel container spark o nlp (solo pandas e scikit-learn).
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from ft_utils import logit, recall_at, weighted_ap, weighted_review_table  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

NLP = Path(os.getenv("NLP_DIR", "data/nlp"))
REPORTS = Path(os.getenv("REPORTS_DIR", "reports"))
MODELS = Path(os.getenv("MODELS_DIR", "models"))
B = int(os.getenv("BOOTSTRAP_B", "200"))
PRED = os.getenv("PRED_FILE", "pred_stage2.parquet")
META = os.getenv("META_FILE", "nlp_stage2_metadata.json")
(REPORTS / "figures").mkdir(parents=True, exist_ok=True)
lines = []


def log(text=""):
    print(text, flush=True)
    lines.append(text)


def block(df):
    log("```text")
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}", "display.width", 200):
        log(df.to_string(index=False))
    log("```")


nar = pd.read_parquet(NLP / "narratives", columns=["odino", "text", "serious", "split", "year"])
s1 = pd.read_parquet(NLP / "stage1_scores.parquet", columns=["odino", "split", "is_candidate"])
pred = pd.read_parquet(NLP / PRED)
meta = json.loads((MODELS / META).read_text(encoding="utf-8"))

te_all = nar[nar["split"] == "test"]
N, P = len(te_all), int(te_all["serious"].sum())
share = float(s1[(s1["split"] == "test")]["is_candidate"].mean())
val = pred[pred["split"] == "validation"].reset_index(drop=True)
te = pred[pred["split"] == "test"].reset_index(drop=True)
y, w = te["serious"].to_numpy(), te["w"].to_numpy()

# ------------------------------------------------------------------ combinazione (si addestra solo sulla validation)
Xv = np.column_stack([logit(val["score1"]), logit(val["score2"])])
Xt = np.column_stack([logit(te["score1"]), logit(te["score2"])])
blend = LogisticRegression(C=1.0).fit(Xv, val["serious"], sample_weight=val["w"])
te["score3"] = blend.predict_proba(Xt)[:, 1]
scores = {"Solo TF-IDF": te["score1"].to_numpy(), "Solo transformer": te["score2"].to_numpy(),
          "Combinazione (LR sulla validation)": te["score3"].to_numpy()}

log("# Milestone 4: confronto della cascata (TF-IDF, transformer, combinazione)")
log()
log(f"Modello: {meta['model']} | lunghezza massima {meta['max_len']} token | addestramento su {meta['n_train']:,} esempi "
    f"({meta['n_train_pos']:,} positivi), {meta['epochs']} epoch, {meta['train_minutes']} min | "
    f"inferenza {meta['inference_docs_per_second']} testi/s su {meta['threads']} thread")
log(f"Test: {N:,} reclami, {P:,} esiti gravi ({P / N:.2%}). Candidati dello stadio 1: {share:.1%} del test. "
    f"Campione pesato: {len(te):,} righe (peso totale {w.sum():,.0f}), bootstrap B={B}.")
log(f"Combinazione: logit = {blend.intercept_[0]:+.2f} {blend.coef_[0][0]:+.2f}*logit(TF-IDF) "
    f"{blend.coef_[0][1]:+.2f}*logit(transformer)")

# ------------------------------------------------------------------ PR-AUC pesata tra i candidati
rng = np.random.default_rng(42)
boot_idx = [rng.integers(0, len(te), len(te)) for _ in range(B)]


def boot(metric):
    return np.array([metric(i) for i in boot_idx])


rows = []
base_ap = boot(lambda i: weighted_ap(y[i], scores["Solo TF-IDF"][i], w[i]))
for name, s in scores.items():
    pt = weighted_ap(y, s, w)
    bs = boot(lambda i, s=s: weighted_ap(y[i], s[i], w[i]))
    d = bs - base_ap
    rows.append({"ranker": name, "pr_auc_candidati": pt, "ic_low": np.percentile(bs, 2.5), "ic_high": np.percentile(bs, 97.5),
                 "diff_vs_tfidf": pt - weighted_ap(y, scores["Solo TF-IDF"], w),
                 "diff_ic_low": np.percentile(d, 2.5), "diff_ic_high": np.percentile(d, 97.5)})
log()
log("## 1. PR-AUC pesata tra i candidati (la regione in cui i due stadi competono)")
log()
block(pd.DataFrame(rows))

# ------------------------------------------------------------------ recall al tasso di revisione
rates = [r for r in (0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20) if r <= share]
tabs = {}
for name, s in scores.items():
    t = weighted_review_table(y, s, w, N, P, rates)
    t.insert(0, "ranker", name)
    tabs[name] = t
all_tab = pd.concat(tabs.values())
log()
log("## 2. Rivedendo la quota indicata di TUTTI i reclami del test: casi gravi catturati (recall) e precisione")
log()
for name, t in tabs.items():
    log(f"{name}:")
    block(t.drop(columns="ranker"))
all_tab.to_csv(REPORTS / "m4_cascade_recall.csv", index=False)

log()
log("## 3. Differenza di recall rispetto al solo TF-IDF, con IC bootstrap al 95% (positivo = meglio del TF-IDF)")
drows = []
for r in [x for x in (0.01, 0.02, 0.05, 0.10) if x <= share]:
    base = np.array([recall_at(y[i], scores["Solo TF-IDF"][i], w[i], N, P, r) for i in boot_idx])
    for name in ("Solo transformer", "Combinazione (LR sulla validation)"):
        s = scores[name]
        bs = np.array([recall_at(y[i], s[i], w[i], N, P, r) for i in boot_idx])
        pt = recall_at(y, s, w, N, P, r) - recall_at(y, scores["Solo TF-IDF"], w, N, P, r)
        drows.append({"review_rate_%": 100 * r, "ranker": name, "delta_recall": pt,
                      "ic_low": np.percentile(bs - base, 2.5), "ic_high": np.percentile(bs - base, 97.5)})
log()
block(pd.DataFrame(drows))

# ------------------------------------------------------------------ robustezza per anno e per stile
SCORE_COL = {"Solo TF-IDF": "score1", "Solo transformer": "score2", "Combinazione (LR sulla validation)": "score3"}
all_te = te_all.copy()
all_te["stile"] = np.where(all_te["text"].str.lower().str.startswith("the contact"), "the contact", "altro")
m = te.merge(nar[["odino", "text", "year"]], on="odino", how="left")
m["stile"] = np.where(m["text"].str.lower().str.startswith("the contact"), "the contact", "altro")
rows = []
for kind, col in (("anno", "year"), ("stile", "stile")):
    for g in sorted(all_te[col].unique()):
        sub, pop = m[m[col] == g], all_te[all_te[col] == g]
        Ng, Pg = len(pop), int(pop["serious"].sum())
        for name, c in SCORE_COL.items():
            rows.append({"gruppo": f"{kind}: {g}", "reclami": Ng, "tasso_grave": Pg / Ng, "ranker": name,
                         "pr_auc_candidati": weighted_ap(sub["serious"].to_numpy(), sub[c].to_numpy(), sub["w"].to_numpy()),
                         "recall_5%": recall_at(sub["serious"].to_numpy(), sub[c].to_numpy(), sub["w"].to_numpy(), Ng, Pg, 0.05)})
log()
log("## 4. Robustezza: per anno del test e per stile della narrativa ('the contact' = scritta da un operatore)")
log("Nel train un prefisso 'TL*' marcava parte dei reclami di operatore; dal 2022 non esiste piu', ma lo stile e' rimasto.")
log("Differenze grandi tra gruppi indicano dove il modello e' meno affidabile e cosa monitorare.")
log()
block(pd.DataFrame(rows))

# ------------------------------------------------------------------ ROI (ipotesi di costo esplicite)
log()
log("## 5. Costo atteso per 1.000 reclami (ipotesi: rileggere un reclamo costa 1; non vedere un caso grave costa V)")
log()
grid = [r for r in np.arange(0.005, share + 1e-9, 0.005)]
curves = {name: weighted_review_table(y, s, w, N, P, grid)["recall"].to_numpy() for name, s in scores.items()}
crow = []
for V in (10, 30, 100):
    no_triage, review_all = 1000 * (P / N) * V, 1000.0
    for name, rec in curves.items():
        cost = 1000 * (np.array(grid) + (P / N) * (1 - rec) * V)
        k = int(np.argmin(cost))
        crow.append({"V": V, "ranker": name, "quota_ottima_%": 100 * grid[k], "recall": rec[k], "costo": cost[k],
                     "senza_triage": no_triage, "rivedere_tutto": review_all,
                     "risparmio_%": 100 * (1 - cost[k] / min(no_triage, review_all))})
block(pd.DataFrame(crow))
log("I costi sono ipotesi, non misure: servono a mostrare come la scelta della quota di revisione dipende dal rapporto "
    "tra il costo di un caso mancato e quello di una rilettura. Da sostituire con i valori dell'azienda.")

dps = meta["inference_docs_per_second"]
log()
log(f"Costo di calcolo: stadio 2 a {dps} testi/s -> {1000 * share / dps / 60:.1f} min per 1.000 reclami (solo candidati) "
    f"contro {1000 / dps / 60:.1f} min se il transformer rileggesse tutto; il TF-IDF su tutti i reclami e' trascurabile.")

# ------------------------------------------------------------------ grafico
fig, ax = plt.subplots(figsize=(6.6, 4.6))
for name, rec in curves.items():
    ax.plot(100 * np.array(grid), rec, marker="o", ms=3, label=name)
ax.set_xlabel("quota di reclami rivista (%)")
ax.set_ylabel("casi gravi catturati (recall)")
ax.set_title("Recall al tasso di revisione, test")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(REPORTS / "figures" / "m4_cascade_recall_curves.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ esempi (solo su disco locale, non nei report)
ex = te.merge(nar[["odino", "text"]], on="odino", how="left")
ex["delta"] = logit(ex["score1"]) - logit(ex["score2"])
fp = ex[ex["serious"] == 0].sort_values("delta", ascending=False).head(5)
fn = ex[ex["serious"] == 1].sort_values("delta", ascending=True).head(5)
out = ["# Esempi (contengono testo dei reclami: non committare)\n", "## Il TF-IDF li riteneva gravi, il transformer no, ed erano non gravi\n"]
out += [f"- tf-idf {r.score1:.3f} | transformer {r.score2:.3f}: {r.text[:300]}\n" for r in fp.itertuples()]
out += ["\n## Il transformer li ha recuperati: gravi, con punteggio TF-IDF relativamente basso\n"]
out += [f"- tf-idf {r.score1:.3f} | transformer {r.score2:.3f}: {r.text[:300]}\n" for r in fn.itertuples()]
(NLP / "error_examples.md").write_text("".join(out), encoding="utf-8")

(REPORTS / "m4_cascade_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nReport scritto in {REPORTS / 'm4_cascade_comparison.md'}")
