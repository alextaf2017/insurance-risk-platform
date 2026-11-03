"""Controlli di data quality minimali: falliscono il job se una regola è violata."""
from pyspark.sql import DataFrame, functions as F


def assert_rules(df: DataFrame, table: str, rules: dict, unique_key: str = None) -> None:
    """rules: nome -> espressione SQL che deve essere vera per ogni riga (NULL = violazione)."""
    aggs = [F.sum(F.when(F.expr(expr), 0).otherwise(1)).alias(name) for name, expr in rules.items()]
    res = df.agg(F.count("*").alias("_rows"), *aggs).first().asDict()
    n_rows = res.pop("_rows")
    failures = {k: v for k, v in res.items() if v}

    if unique_key:
        dup = n_rows - df.select(unique_key).distinct().count()
        if dup:
            failures[f"unique({unique_key})"] = dup

    n_rules = len(rules) + int(bool(unique_key))
    print(f"[DQ] {table}: {n_rows:,} righe, {n_rules} regole")
    if failures:
        raise AssertionError(f"[DQ] {table} violazioni: {failures}")
    print(f"[DQ] {table}: OK")
