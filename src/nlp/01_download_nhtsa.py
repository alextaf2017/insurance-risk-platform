"""Scarica i reclami NHTSA (dominio pubblico) e li converte in un Parquet ridotto.

Ordine di tentativi:
1. file gia' presenti in data/raw/nhtsa/ (zip o txt): utile se si scarica a mano dal sito NHTSA;
2. il file completo FLAT_CMPL.zip (circa 400 MB), da due indirizzi noti;
3. solo con NHTSA_USE_CHUNKS=1: gli zip a blocchi di 5 anni (COMPLAINTS_RECEIVED_YYYY-YYYY.zip). Il blocco
   corrente (2025-2029) risulta non disponibile (404), quindi i blocchi perdono i reclami piu' recenti.
Gli indirizzi si possono forzare con la variabile NHTSA_URLS (separati da virgola).

Variabili: RAW_DIR (default data/raw), NHTSA_FROM_YEAR (default 2018).
"""
import os
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

from nhtsa_parse import convert_to_parquet

RAW = Path(os.getenv("RAW_DIR", "data/raw")) / "nhtsa"
FROM_YEAR = int(os.getenv("NHTSA_FROM_YEAR", "2018"))
BASE = "https://static.nhtsa.gov/odi/ffdd/cmpl/"
FULL = [BASE + "FLAT_CMPL.zip", "http://www-odi.nhtsa.dot.gov/downloads/folders/Complaints/FLAT_CMPL.zip"]
UA = {"User-Agent": "Mozilla/5.0 (insurance-risk-platform; research use)"}


def chunk_urls(from_year: int) -> list:
    """Blocchi di 5 anni (2015-2019, 2020-2024, ...) che coprono da from_year a oggi."""
    start = (from_year // 5) * 5
    return [f"{BASE}COMPLAINTS_RECEIVED_{y}-{y + 4}.zip" for y in range(start, date.today().year + 1, 5)]


def download(url: str, dest: Path, retries: int = 2) -> bool:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done, t0 = 0, time.time()
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
                    done += len(block)
                    if done % (50 << 20) < (1 << 20):
                        print(f"  {done / 1e6:,.0f} MB" + (f" / {total / 1e6:,.0f} MB" if total else "")
                              + f" ({time.time() - t0:.0f}s)", flush=True)
            print(f"[ok] {url} -> {dest} ({dest.stat().st_size / 1e6:,.0f} MB)")
            return True
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"[errore] {url}: {e} (tentativo {attempt + 1}/{retries + 1})")
            dest.unlink(missing_ok=True)
            time.sleep(2)
    return False


def try_urls(urls: list) -> list:
    """Scarica tutti gli indirizzi; se uno fallisce annulla e restituisce una lista vuota."""
    got = []
    for u in urls:
        dest = RAW / u.rsplit("/", 1)[-1]
        if not download(u, dest):
            for p in got:
                p.unlink(missing_ok=True)
            return []
        got.append(dest)
    return got


RAW.mkdir(parents=True, exist_ok=True)
local = sorted(p for p in RAW.iterdir() if p.suffix.lower() in (".zip", ".txt"))

if local:
    print(f"[locale] uso i file gia' presenti: {[p.name for p in local]}")
else:
    custom = [u.strip() for u in os.getenv("NHTSA_URLS", "").split(",") if u.strip()]
    if custom:
        local = try_urls(custom)
    else:
        if os.getenv("NHTSA_USE_CHUNKS", "0") == "1":
            local = try_urls(chunk_urls(FROM_YEAR))
        for url in FULL:
            if local:
                break
            local = try_urls([url])
    if not local:
        raise SystemExit(
            "Download non riuscito. Scarica a mano FLAT_CMPL.zip (o gli zip a blocchi di 5 anni) dalla pagina "
            "NHTSA Datasets and APIs, copiali in data/raw/nhtsa/ e rilancia."
        )

out = RAW / "complaints.parquet"
print(f"[parse] conversione in {out} (anni >= {FROM_YEAR})")
info = convert_to_parquet(local, out, from_year=FROM_YEAR)
print(f"[ok] righe tenute: {info['rows_kept']:,} | file letti: {info['files']} | codifica: {info['encodings']}")
if info["rows_kept"] == 0:
    raise SystemExit("Nessuna riga nel periodo richiesto: controlla NHTSA_FROM_YEAR e il formato del file.")
