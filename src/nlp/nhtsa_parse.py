"""Lettura del file piatto dei reclami NHTSA (CMPL.txt: 51 campi separati da tabulazione, senza intestazione).

Minimizzazione dei dati: si tengono solo i campi necessari al progetto. Restano fuori citta', VIN, dati del
concessionario e nome dell'operatore del veicolo (campi 13, 15, 41-45, 51), cioe' tutto cio' che puo'
identificare una persona o un'azienda. Il testo libero (CDESCR) puo' comunque contenere dati personali che
NHTSA non ha oscurato: va trattato come dato sensibile anche se pubblico.
"""
import csv
import io
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Ordine dei campi secondo il dizionario dati (https://static.nhtsa.gov/odi/ffdd/cmpl/CMPL.txt).
# I campi 50-51 sono stati aggiunti ad aprile 2026: i file piu' vecchi ne hanno 49.
COLUMNS = [
    "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CRASH", "FAILDATE", "FIRE", "INJURED",
    "DEATHS", "COMPDESC", "CITY", "STATE", "VIN", "DATEA", "LDATE", "MILES", "OCCURENCES", "CDESCR",
    "CMPL_TYPE", "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN", "ANTI_BRAKES_YN", "CRUISE_CONT_YN", "NUM_CYLS",
    "DRIVE_TRAIN", "FUEL_SYS", "FUEL_TYPE", "TRANS_TYPE", "VEH_SPEED", "DOT", "TIRE_SIZE", "LOC_OF_TIRE",
    "TIRE_FAIL_TYPE", "ORIG_EQUIP_YN", "MANUF_DT", "SEAT_TYPE", "RESTRAINT_TYPE", "DEALER_NAME", "DEALER_TEL",
    "DEALER_CITY", "DEALER_STATE", "DEALER_ZIP", "PROD_TYPE", "REPAIRED_YN", "MEDICAL_ATTN", "VEHICLES_TOWED_YN",
    "STATE_OF_INCIDENT", "VEHICLE_OPERATOR",
]
assert len(COLUMNS) == 51

KEEP = [
    "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CRASH", "FAILDATE", "FIRE", "INJURED",
    "DEATHS", "COMPDESC", "LDATE", "CDESCR", "CMPL_TYPE", "POLICE_RPT_YN", "PROD_TYPE", "MEDICAL_ATTN",
    "VEHICLES_TOWED_YN",
]


def detect_encoding(stream, probe_bytes: int = 4_000_000) -> str:
    """UTF-8 se i primi MB lo sono, altrimenti cp1252 (tipico dei file di origine Windows)."""
    head = stream.read(probe_bytes)
    cut = head.rfind(b"\n")
    head = head[: cut if cut > 0 else len(head)]
    try:
        head.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp1252"


def _open_text_member(path: Path):
    """Restituisce una funzione che apre ogni file di testo contenuto nel percorso (zip o txt)."""
    if path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        names = [n for n in zf.namelist() if n.lower().endswith((".txt", ".lst"))]
        return [(n, lambda n=n, zf=zf: zf.open(n)) for n in names]
    return [(path.name, lambda: open(path, "rb"))]


def iter_chunks(opener, from_year: int, chunksize: int = 200_000):
    """Legge un file a blocchi, tiene le colonne utili e filtra per anno di ricezione (LDATE >= from_year)."""
    with opener() as probe:
        enc = detect_encoding(probe)
    with opener() as stream:
        reader = pd.read_csv(
            stream, sep="\t", header=None, names=COLUMNS, usecols=KEEP, dtype=str, keep_default_na=False,
            quoting=csv.QUOTE_NONE, encoding=enc, encoding_errors="replace", on_bad_lines="skip",
            chunksize=chunksize,
        )
        limit = f"{from_year}0101"
        for chunk in reader:
            yield enc, chunk[chunk["LDATE"] >= limit].reset_index(drop=True)


def convert_to_parquet(inputs, out_path: Path, from_year: int = 2018) -> dict:
    """Converte uno o piu' file (zip o txt) in un unico Parquet con le sole colonne di KEEP."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer, n_files, rows_kept, encodings = None, 0, 0, set()
    for path in inputs:
        for name, opener in _open_text_member(Path(path)):
            for enc, chunk in iter_chunks(opener, from_year):
                encodings.add(enc)
                rows_kept += len(chunk)
                if len(chunk) == 0:
                    continue
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
                writer.write_table(table)
            n_files += 1
    if writer is not None:
        writer.close()
    return {"rows_kept": rows_kept, "files": n_files, "encodings": sorted(encodings)}


def sniff_layout(path: Path, n: int = 3) -> list:
    """Utile per diagnosi: numero di campi delle prime righe di un file di testo."""
    members = _open_text_member(path)
    name, opener = members[0]
    with opener() as f:
        lines = [io.BytesIO(f.readline()).getvalue() for _ in range(n)]
    return [len(line.split(b"\t")) for line in lines]
