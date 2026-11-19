"""Funzioni del business case (milestone 7): selezione avversa in un mercato a due assicuratori, VAN e punto di pareggio.

Python 3.8 compatibile (gira nel container spark).

Idea del modello di mercato: ogni polizza del test e' un cliente. Due assicuratori, H (modello nuovo) e G (modello di
confronto), prezzano la stessa polizza con le proprie previsioni. Ciascuno RIBASA il livello dei prezzi in modo che, se
ottenesse l'intero portafoglio medio, il rapporto sinistri/premi (loss ratio) fosse esattamente quello obiettivo: cosi'
il livello (coda, sinistri senza importo ...) si cancella e conta solo la differenza di RELATIVITA' tra i prezzi.
Una quota `shop` di clienti confronta i prezzi e sceglie H con probabilita' sigmoide(-beta * ln(P_H / P_G)); gli altri
restano con un assicuratore scelto a caso (50/50). Se H prezza meglio, attira i clienti che G sovrapprezza e lascia a G
quelli che G sottoprezza (selezione avversa): il guadagno di H e' la perdita di G.
Il costo di ogni polizza e' quello REALE osservato nel test (importi con cap, eventuali sinistri senza importo).
"""
from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rebase(model_premium, cost, loss_ratio_target):
    """Premi (per polizza) con livello tale che sum(premi) = sum(costi) / loss_ratio_target."""
    model_premium = np.asarray(model_premium, dtype=float)
    return model_premium * (np.sum(cost) / loss_ratio_target) / model_premium.sum()


def market(prem_h, prem_g, cost, expo, shop, beta, expense_ratio, loss_ratio_target):
    """Esito atteso per i due assicuratori (probabilita' di scelta, costi reali). Restituisce un dizionario.

    Il confronto e' con il caso base in cui H e G usano lo stesso modello: ciascuno ha meta' del mercato e un loss ratio
    pari all'obiettivo. `delta_h` e `delta_g` sono la variazione del risultato tecnico (premi x (1 - spese) - sinistri)
    di H e di G rispetto a quel caso base, per anno-polizza del mercato e in % del premio totale del caso base.
    Un modello preciso guadagna poco (resta al margine obiettivo su qualunque polizza acquisisca: ottiene piu' volume);
    il costo vero lo paga chi prezza peggio, che eredita le polizze sottoprezzate: e' il valore DIFENSIVO di un buon modello.
    """
    prem_h, prem_g = np.asarray(prem_h, float), np.asarray(prem_g, float)
    cost, expo = np.asarray(cost, float), np.asarray(expo, float)
    w = (1.0 - shop) * 0.5 + shop * sigmoid(-beta * np.log(prem_h / prem_g))  # quota di ogni polizza che va a H
    tot_expo, tot_cost = float(expo.sum()), float(cost.sum())
    base_prem = tot_cost / loss_ratio_target
    base_result = base_prem * (1 - expense_ratio) - tot_cost  # risultato dell'intero mercato nel caso base
    out = {"premium_per_py": base_prem / tot_expo, "cost_per_py": tot_cost / tot_expo}
    for name, share, prem in (("h", w, prem_h), ("g", 1.0 - w, prem_g)):
        p, c, e = float((share * prem).sum()), float((share * cost).sum()), float((share * expo).sum())
        result = p * (1 - expense_ratio) - c
        out[name] = {"premium": p, "cost": c, "exposure": e, "loss_ratio": c / p, "share": e / tot_expo,
                     "delta": (result - base_result / 2) / tot_expo,
                     "delta_pct_premium": 100 * (result - base_result / 2) / base_prem}
    return out


def market_bootstrap(rate_h, rate_g, cost, expo, shop, beta, expense_ratio, lr_target, B=200, seed=0):
    """Effetto su H e su G (in % del premio del caso base) con IC bootstrap al 95%, ricampionando le polizze.
    I costi sono quelli osservati: con pochi sinistri il rumore e' grande e l'IC lo mostra."""
    rate_h, rate_g = np.asarray(rate_h, float), np.asarray(rate_g, float)
    cost, expo = np.asarray(cost, float), np.asarray(expo, float)
    rng = np.random.default_rng(seed)
    n = len(cost)

    def one(idx):
        ph = rebase(rate_h[idx] * expo[idx], cost[idx], lr_target)
        pg = rebase(rate_g[idx] * expo[idx], cost[idx], lr_target)
        return market(ph, pg, cost[idx], expo[idx], shop, beta, expense_ratio, lr_target)

    full = one(np.arange(n))
    dh, dg = [], []
    for _ in range(B):
        r = one(rng.integers(0, n, n))
        dh.append(r["h"]["delta_pct_premium"])
        dg.append(r["g"]["delta_pct_premium"])
    q = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]  # noqa: E731
    return {"delta_h_pct": full["h"]["delta_pct_premium"], "delta_h_ci": q(dh),
            "delta_g_pct": full["g"]["delta_pct_premium"], "delta_g_ci": q(dg),
            "delta_h_per_py": full["h"]["delta"], "delta_g_per_py": full["g"]["delta"],
            "loss_ratio_h": full["h"]["loss_ratio"], "loss_ratio_g": full["g"]["loss_ratio"], "share_h": full["h"]["share"],
            "premium_per_py": full["premium_per_py"], "cost_per_py": full["cost_per_py"]}


def npv(flows, rate):
    """Valore attuale netto: flows[0] e' all'anno 0 (non scontato)."""
    return float(sum(f / (1 + rate) ** t for t, f in enumerate(flows)))


def payback_year(flows):
    """Primo anno in cui il flusso cumulato (non scontato) diventa >= 0; None se non succede."""
    cum = 0.0
    for t, f in enumerate(flows):
        cum += f
        if cum >= 0:
            return t
    return None


def break_even_book(costs, gain_per_py, rate):
    """Anni-polizza del portafoglio per cui il VAN e' zero: VA(costi) / VA(guadagno unitario).
    costs[t] e gain_per_py[t] hanno la stessa lunghezza (anno 0..T); None se il guadagno attualizzato non e' positivo."""
    pv_cost = sum(c / (1 + rate) ** t for t, c in enumerate(costs))
    pv_gain = sum(g / (1 + rate) ** t for t, g in enumerate(gain_per_py))
    return None if pv_gain <= 0 else float(pv_cost / pv_gain)


def cost_schedule(costs: dict, mult: float = 1.0, horizon: int = 3):
    """Flussi di costo (negativi) per gli anni 0..horizon: sviluppo e validazione all'anno 0, poi manutenzione e infrastruttura."""
    fte = costs["fte_year_cost"]
    year0 = costs["build_fte"] * costs["build_months"] / 12 * fte + costs["validation_fte_once"] * fte
    run = costs["run_fte_per_year"] * fte + costs["infra_per_year"]
    return [-mult * year0] + [-mult * run] * horizon


def benefit_schedule(delta_per_market_py: float, book_py: float, level_factor: float, shrink: float, decay, horizon: int = 3):
    """Benefici annui (anni 0..horizon; l'anno 0 e' di sviluppo e non ne ha).
    `delta_per_market_py`: effetto misurato per anno-polizza del mercato simulato (che ha esposizione doppia del portafoglio
    proprio: H e G hanno ciascuno meta'); `level_factor`: fattore di scala dei costi (coda) rispetto al costo con cap;
    `shrink`: quota del guadagno che si conserva in produzione; `decay`: quota mantenuta negli anni 1..horizon."""
    per_year = delta_per_market_py * 2.0 * book_py * level_factor * shrink
    return [0.0] + [per_year * decay[t] for t in range(horizon)]
