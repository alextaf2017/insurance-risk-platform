"""Scarica freMTPL2 (frequenza e severita) da OpenML e lo salva in data/raw come Parquet.

Fonte alternativa: Kaggle, cerca "freMTPL2freq" e "freMTPL2sev"; salva i file come
data/raw/fremtpl2_freq.parquet e data/raw/fremtpl2_sev.parquet.
"""
import os
from pathlib import Path

from sklearn.datasets import fetch_openml

RAW = Path(os.getenv("RAW_DIR", "data/raw"))
DATASETS = {"fremtpl2_freq": 41214, "fremtpl2_sev": 41215}

RAW.mkdir(parents=True, exist_ok=True)
for name, data_id in DATASETS.items():
    out = RAW / f"{name}.parquet"
    if out.exists():
        print(f"[skip] {out} esiste gia")
        continue
    df = fetch_openml(data_id=data_id, as_frame=True, parser="auto", data_home=str(RAW / ".openml")).frame
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)
    df.to_parquet(out, index=False)
    print(f"[ok] {out}: {len(df):,} righe, colonne={list(df.columns)}")
