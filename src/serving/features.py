"""Dal record grezzo della polizza alle feature del modello, con le stesse regole del livello gold del lake.

Regole gold (src/ingestion/04_gold.py): eta' veicolo limitata a 20, eta' conducente a 90, Bonus-Malus a 150,
densita' in logaritmo naturale. Le categoriche si abbinano ai livelli visti dal modello ignorando maiuscole,
punteggiatura e apici (nel silver `veh_gas` puo' contenere apici letterali: vedi i problemi noti).
"""
import math
import re

import pandas as pd

CAPS = {"veh_age": 20, "driv_age": 90, "bonus_malus": 150}


def norm(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def match_level(value, levels) -> tuple:
    """(livello del modello, trovato?). Se non c'e' corrispondenza restituisce il valore originale."""
    key = norm(value)
    for lv in levels:
        if norm(lv) == key:
            return lv, True
    return value, False


def build_frame(policies: list, numeric: list, categorical: list, levels: dict) -> tuple:
    """DataFrame con colonne [categoriche + numeriche] nell'ordine del modello, e un elenco di avvisi per riga."""
    rows, warns = [], []
    for p in policies:
        d = p if isinstance(p, dict) else p.model_dump()
        w = []
        capped = {}
        for k, cap in CAPS.items():
            capped[k] = min(d[k], cap)
            if d[k] > cap:
                w.append(f"{k}={d[k]} limitato a {cap} (stessa regola del livello gold)")
        row = {"driv_age_c": float(capped["driv_age"]), "veh_age_c": float(capped["veh_age"]),
               "bonus_malus_c": float(capped["bonus_malus"]), "veh_power": float(d["veh_power"]),
               "log_density": math.log(d["density"])}
        for c in categorical:
            lv, ok = match_level(d[c], levels.get(c, []))
            row[c] = lv
            if not ok:
                w.append(f"{c}='{d[c]}' non visto in addestramento: trattato come valore mancante")
        rows.append(row)
        warns.append(w)
    df = pd.DataFrame(rows)[categorical + numeric]
    return df, warns
