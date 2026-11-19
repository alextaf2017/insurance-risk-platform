"""Riproduce traffico realistico sull'API (per la demo di monitoraggio). Gira nel container spark (rete compose: http://api:8000).

Le polizze sono prese dal test e i reclami dal test NHTSA: non sono dati inventati. Con --shift aumenta il Bonus-Malus di
x1,5 per una quota --shift-share delle polizze (default 1 = tutte: spostamento molto severo; il PSI oltre 1 e' saturato
dal troncamento a 1e-6 e va letto solo come "azione"). Non e' lo scenario della milestone 5, che ripesa il gruppo con Bonus-Malus > 75.
Uso: python3 src/serving/demo_traffic.py --n 800 [--shift] [--triage 300]
"""
import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = os.getenv("API_URL", "http://api:8000")
MODELING = Path(os.getenv("MODELING_DIR", "data/modeling"))
NLP = Path(os.getenv("NLP_DIR", "data/nlp"))


def call(path, body=None):
    req = urllib.request.Request(API + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read()), (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "detail": e.read().decode()[:300]}, (time.perf_counter() - t0) * 1000


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="richieste di premio")
    ap.add_argument("--triage", type=int, default=0, help="richieste di triage")
    ap.add_argument("--shift", action="store_true", help="Bonus-Malus x1,5 (scenario di deriva)")
    ap.add_argument("--shift-share", type=float, default=1.0, help="quota di polizze interessate dallo shift (0-1)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ready = None
    for _ in range(60):  # l'avvio con PyTorch puo' richiedere un minuto
        try:
            ready, _ = call("/ready")
            break
        except urllib.error.URLError:
            time.sleep(3)
    if ready is None:
        raise SystemExit(f"API non raggiungibile su {API}: controlla `make api-logs` e `docker compose ps`")
    print("[ready]", json.dumps(ready))

    pol = pd.read_parquet(MODELING / "policies")
    pol = pol[pol["split"] == "test"].sample(a.n, random_state=a.seed)
    lat, errors = [], 0
    rng = np.random.default_rng(a.seed)
    hit = rng.random(len(pol)) < a.shift_share
    for r, h in zip(pol.itertuples(), hit):
        bm = int(round(r.bonus_malus_c * (1.5 if (a.shift and h) else 1.0)))
        body = {"driv_age": int(r.driv_age_c), "veh_age": int(r.veh_age_c), "bonus_malus": max(50, min(bm, 350)),
                "veh_power": int(r.veh_power), "density": float(math.exp(r.log_density)), "veh_gas": str(r.veh_gas),
                "veh_brand": str(r.veh_brand), "region": str(r.region), "exposure": float(min(max(r.exposure, 1e-3), 1.0))}
        out, ms = call("/v1/premium", body)
        errors += "http_error" in out
        lat.append(ms)
    print(f"[premio] {a.n} richieste, errori {errors} | latenza ms: p50 {pct(lat, 50):.1f} p95 {pct(lat, 95):.1f} max {max(lat):.1f}")

    if a.triage:
        nar = pd.read_parquet(NLP / "narratives", columns=["text", "split"])
        nar = nar[nar["split"] == "test"].sample(a.triage, random_state=a.seed)
        lat, errors, cascade = [], 0, 0
        for t in nar["text"]:
            out, ms = call("/v1/triage", {"text": t})
            errors += "http_error" in out
            cascade += out.get("route") == "cascade"
            lat.append(ms)
        print(f"[triage] {a.triage} richieste, errori {errors}, passate allo stadio 2: {cascade} | "
              f"latenza ms: p50 {pct(lat, 50):.0f} p95 {pct(lat, 95):.0f} max {max(lat):.0f}")

    # finestra = solo il traffico di questa esecuzione: il log e' cumulativo e mescolare le esecuzioni falserebbe il confronto
    s, _ = call(f"/monitor/summary?last_premium={a.n}&last_triage={max(a.triage, 1)}")
    print("\n[monitoraggio] (solo il traffico di questa esecuzione)")
    print(json.dumps(s, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
