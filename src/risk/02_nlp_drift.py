"""Milestone 5B - Drift dei testi NHTSA anno per anno e verifica delle soglie di riaddestramento.

Il TF-IDF di stadio 1 (che ha punteggi su tutti i reclami, fuori campione per il train) e le caratteristiche dei testi
permettono di misurare, anno per anno, quanto si muove la popolazione e quanto il modello ne risente.
Si valutano le soglie proposte nella model card:
  - quota di candidati (soglia fissa fissata sulla validation) oltre +-5 punti dal 20%
  - lunghezza media delle narrative oltre 140 parole
  - recall al 5% di revisione dello stadio 1 inferiore di piu' di 5 punti al valore del 2024 (anno di validation)
e si confronta il PSI (convenzione: > 0,10 moderato, > 0,25 forte) con l'effetto reale sulle prestazioni.

Richiede 05_stage1_scores.py. Output: reports/m5_nlp_drift.md, reports/m5_nlp_drift.csv, reports/figures/m5_nlp_drift.png.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "nlp"))
sys.path.insert(0, str(HERE))
from nlp_metrics import review_rate_table  # noqa: E402
from risk_lib import psi  # noqa: E402

NLP = Path(os.getenv("NLP_DIR", "data/nlp"))
REPORTS = Path(os.getenv("REPORTS_DIR", "reports"))
(REPORTS / "figures").mkdir(parents=True, exist_ok=True)
CAND_TOL, LEN_MAX, RECALL_DROP = 0.05, 140.0, 0.05
LONG_WORDS = 190  # circa 256 token (circa 1,35 token per parola)
lines = []


def log(text=""):
    print(text, flush=True)
    lines.append(text)


def block(df):
    log("```text")
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}", "display.width", 220):
        log(df.to_string(index=False))
    log("```")


nar = pd.read_parquet(NLP / "narratives", columns=["odino", "text", "year", "split", "serious", "n_words"])
s1 = pd.read_parquet(NLP / "stage1_scores.parquet", columns=["odino", "score1"])
meta = json.loads((NLP / "stage1_meta.json").read_text(encoding="utf-8"))
thr, q = meta["thr_validation"], meta["candidate_q"]
d = nar.merge(s1, on="odino", how="inner")
low = d["text"].str.lower()
d["the_contact"] = low.str.startswith("the contact")
d["prima_persona"] = low.str.match(r"^(i|we|my) ")
d["lungo"] = d["n_words"] > LONG_WORDS
d["candidato"] = d["score1"] >= thr
ref = d[d["split"] == "train"]
years = sorted(d["year"].unique())

rows, prev = [], None
for y in years:
    g = d[d["year"] == y]
    yy, ss = g["serious"].to_numpy(), g["score1"].to_numpy()
    r5 = float(review_rate_table(yy, ss, rates=[0.05])["recall"].iloc[0])
    rows.append({
        "anno": int(y), "reclami": len(g), "tasso_grave": yy.mean(), "parole_medie": g["n_words"].mean(),
        "testi_lunghi_%": 100 * g["lungo"].mean(), "the_contact_%": 100 * g["the_contact"].mean(),
        "prima_persona_%": 100 * g["prima_persona"].mean(),
        "psi_punteggio_vs_train": psi(ref["score1"], ss), "psi_parole_vs_train": psi(ref["n_words"], g["n_words"]),
        "psi_punteggio_vs_prec": psi(prev["score1"], ss) if prev is not None else np.nan,
        "psi_parole_vs_prec": psi(prev["n_words"], g["n_words"]) if prev is not None else np.nan,
        "candidati_%": 100 * g["candidato"].mean(), "gravi_nei_candidati_%": 100 * g.loc[g["candidato"], "serious"].sum() / max(1, yy.sum()),
        "pr_auc_tfidf": float(average_precision_score(yy, ss)), "recall_5%": r5,
        "prob_media_prevista": ss.mean(), "prev_su_oss": ss.mean() / yy.mean(),
    })
    prev = g
tab = pd.DataFrame(rows)
tab.to_csv(REPORTS / "m5_nlp_drift.csv", index=False)

ref_recall = float(tab.loc[tab["anno"] == 2024, "recall_5%"].iloc[0]) if 2024 in set(tab["anno"]) else float(tab["recall_5%"].median())
tab["scatta_candidati"] = (tab["candidati_%"] / 100 - q).abs() > CAND_TOL
tab["scatta_lunghezza"] = tab["parole_medie"] > LEN_MAX
tab["scatta_recall"] = tab["recall_5%"] < ref_recall - RECALL_DROP
tab["psi>0.10"] = (tab["psi_punteggio_vs_train"] > 0.10) | (tab["psi_parole_vs_train"] > 0.10)

log("# Milestone 5B: drift dei testi NHTSA anno per anno")
log()
log(f"Stadio 1 (TF-IDF): soglia fissa dei candidati {thr:.5f} (top {q:.0%} sulla validation). Punteggi fuori campione per gli anni di train "
    f"(2018-2023), del modello addestrato sul train per validation (2024) e test (2025-2026). Il 2026 e' un anno parziale. "
    f"Riferimento del PSI: tutto il train. 'Testi lunghi' = oltre {LONG_WORDS} parole (circa 256 token).")
log()
log("## 1. Indicatori per anno")
log()
block(tab[["anno", "reclami", "tasso_grave", "parole_medie", "testi_lunghi_%", "the_contact_%", "prima_persona_%"]])
log()
log("## 2. Stabilita' del punteggio e delle prestazioni")
log()
block(tab[["anno", "psi_punteggio_vs_train", "psi_punteggio_vs_prec", "psi_parole_vs_train", "psi_parole_vs_prec", "candidati_%",
           "gravi_nei_candidati_%", "pr_auc_tfidf", "recall_5%", "prob_media_prevista", "prev_su_oss"]])
log("`prev_su_oss` = probabilita' media prevista / tasso grave osservato (1 = nessuna deriva del livello); `candidati_%` usa la soglia fissa.")
log()
log(f"## 3. Le soglie di riaddestramento proposte nella model card (recall al 5% di riferimento: {ref_recall:.3f}, anno 2024)")
log()
block(tab[["anno", "candidati_%", "scatta_candidati", "parole_medie", "scatta_lunghezza", "recall_5%", "scatta_recall", "psi>0.10"]])
fired = tab[["scatta_candidati", "scatta_lunghezza", "scatta_recall"]].any(axis=1)
log(f"Anni in cui scatterebbe almeno una soglia: {', '.join(str(a) for a in tab.loc[fired, 'anno']) or 'nessuno'}. "
    f"Anni con PSI > 0,10: {', '.join(str(a) for a in tab.loc[tab['psi>0.10'], 'anno']) or 'nessuno'}.")
log("Lettura: una soglia utile scatta prima che le prestazioni peggiorino; se il PSI e' alto ma le prestazioni non cambiano, "
    "la soglia e' troppo sensibile; se le prestazioni calano senza che nessuna soglia scatti, ne manca una.")

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
x = tab["anno"]
ax[0, 0].plot(x, tab["parole_medie"], "o-", label="parole medie")
ax[0, 0].axhline(LEN_MAX, color="red", ls="--", lw=1)
ax[0, 0].set_title("Lunghezza delle narrative (soglia 140)")
ax[0, 1].plot(x, tab["candidati_%"], "o-")
ax[0, 1].axhspan(100 * (q - CAND_TOL), 100 * (q + CAND_TOL), color="green", alpha=0.15)
ax[0, 1].set_title("Quota di candidati (banda 20% +- 5)")
ax[1, 0].plot(x, tab["pr_auc_tfidf"], "o-", label="PR-AUC")
ax[1, 0].plot(x, tab["recall_5%"], "s-", label="recall al 5%")
ax[1, 0].axhline(ref_recall - RECALL_DROP, color="red", ls="--", lw=1)
ax[1, 0].legend(fontsize=8)
ax[1, 0].set_title("Prestazioni del TF-IDF per anno")
ax[1, 1].plot(x, tab["psi_punteggio_vs_train"], "o-", label="punteggio")
ax[1, 1].plot(x, tab["psi_parole_vs_train"], "s-", label="lunghezza")
ax[1, 1].axhline(0.10, color="grey", ls="--", lw=1)
ax[1, 1].legend(fontsize=8)
ax[1, 1].set_title("PSI rispetto al train (soglia 0,10)")
fig.tight_layout()
fig.savefig(REPORTS / "figures" / "m5_nlp_drift.png", dpi=130)
plt.close(fig)

(REPORTS / "m5_nlp_drift.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nReport scritto in {REPORTS / 'm5_nlp_drift.md'}")
