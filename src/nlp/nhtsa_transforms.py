"""Trasformazioni silver e gold per i reclami NHTSA (funzioni pure su DataFrame, testabili senza catalogo)."""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

# Oscuramento FOIA inserito da NHTSA: e' un artefatto, non contenuto della narrativa.
FOIA_RE = r"(?i)information redacted pursuant to the freedom of information act \(foia\),?\s*5 u\.?s\.?c\.? ?552\(b\)\(6\)\.?"
# Spazi unicode (incluso lo spazio non separabile) che `\s` di Java non riconosce.
UNICODE_SPACES_RE = r"[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF]"
# Sigle dell'analista in coda alla narrativa (es. "*TR", anche piu' di una: "*TT *DT"): firme, non contenuto.
INITIALS_RE = r"(\s*\*[A-Za-z]{2,3}\.?)+\s*$"
INITIALS_TEST_RE = r"\*[A-Za-z]{2,3}\.?\s*$"
# Etichetta di origine in testa alla narrativa (es. "TL* THE CONTACT OWNS ...").
PREFIX_RE = r"^[A-Za-z]{1,4}\*+\s*"


def clean_narrative(col):
    """Ripulisce la narrativa. Restituisce (testo, ha_sigle_in_coda, ha_etichetta_in_testa)."""
    t0 = F.regexp_replace(col, UNICODE_SPACES_RE, " ")
    t1 = F.regexp_replace(t0, FOIA_RE, "[REDACTED]")
    t2 = F.trim(F.regexp_replace(t1, r"\s+", " "))
    t3 = F.trim(F.regexp_replace(t2, INITIALS_RE, ""))
    t4 = F.trim(F.regexp_replace(t3, PREFIX_RE, ""))
    return t4, t2.rlike(INITIALS_TEST_RE).cast("int"), t3.rlike(r"^[A-Za-z]{1,4}\*").cast("int")


def _flag(col: str):
    """'Y' -> 1, tutto il resto (incluso il vuoto, che dal 2021 NHTSA mostra come 'N') -> 0."""
    return (F.upper(F.trim(F.col(col))) == "Y").cast("int")


def _int0(col: str):
    return F.coalesce(F.trim(F.col(col)).cast("double").cast("int"), F.lit(0))


def to_silver(b: DataFrame) -> DataFrame:
    """Un record per reclamo-componente: tipi corretti, solo veicoli, narrativa ripulita."""
    text, had_initials, had_prefix = clean_narrative(F.col("CDESCR"))
    year = F.trim(F.col("YEARTXT")).cast("int")

    s = b.select(
        F.trim(F.col("CMPLID")).cast("long").alias("cmplid"),
        F.trim(F.col("ODINO")).cast("long").alias("odino"),
        F.trim("MFR_NAME").alias("mfr_name"),
        F.upper(F.trim("MAKETXT")).alias("make"),
        F.upper(F.trim("MODELTXT")).alias("model"),
        F.when(year.between(1950, 2100), year).alias("model_year"),  # 9999 = ignoto
        _flag("CRASH").alias("crash"),
        _flag("FIRE").alias("fire"),
        _int0("INJURED").alias("injured"),
        _int0("DEATHS").alias("deaths"),
        _flag("MEDICAL_ATTN").alias("medical_attn"),
        _flag("VEHICLES_TOWED_YN").alias("towed"),
        _flag("POLICE_RPT_YN").alias("police_rpt"),
        F.trim("COMPDESC").alias("compdesc"),
        F.to_date(F.trim(F.col("FAILDATE")), "yyyyMMdd").alias("faildate"),
        F.to_date(F.trim(F.col("LDATE")), "yyyyMMdd").alias("ldate"),
        F.upper(F.trim("CMPL_TYPE")).alias("cmpl_type"),
        F.upper(F.trim("PROD_TYPE")).alias("prod_type"),
        text.alias("text"),
        had_initials.alias("had_initials"),
        had_prefix.alias("had_prefix"),
        F.col("_batch_id"),
    )
    return (
        s.filter(
            (F.col("prod_type") == "V")
            & F.col("cmplid").isNotNull()
            & F.col("odino").isNotNull()
            & F.col("ldate").isNotNull()
            & (F.length("text") >= 30)
        )
        .dropDuplicates(["cmplid"])
    )


def to_gold(s: DataFrame, train_end: str = "2023-12-31", val_end: str = "2024-12-31", min_words: int = 5) -> DataFrame:
    """Un record per reclamo (ODINO): componenti aggregati, etichetta, deduplica del testo e split temporale.

    Etichetta `serious` = almeno una persona ferita, un decesso o cure mediche. I flag sono dichiarati da chi
    presenta il reclamo e non verificati; un 'no' vale 'non indicato'.
    Lo split e' per data di ricezione (train <= train_end < validation <= val_end < test): la valutazione imita
    l'uso reale, in cui il modello vede solo reclami successivi a quelli di addestramento.
    """
    first = lambda c: F.min_by(F.col(c), F.col("cmplid"))  # noqa: E731
    g = s.groupBy("odino").agg(
        F.min("cmplid").alias("cmplid"),
        F.min("ldate").alias("ldate"),
        F.min("faildate").alias("faildate"),
        first("mfr_name").alias("manufacturer"),
        first("make").alias("make"),
        first("model").alias("model"),
        first("model_year").alias("model_year"),
        first("text").alias("text"),
        first("cmpl_type").alias("cmpl_type"),
        F.concat_ws("; ", F.sort_array(F.collect_set("compdesc"))).alias("components"),
        F.size(F.collect_set("compdesc")).alias("n_components"),
        F.max("injured").alias("injured"),
        F.max("deaths").alias("deaths"),
        F.max("medical_attn").alias("medical_attn"),
        F.max("crash").alias("crash"),
        F.max("fire").alias("fire"),
        F.max("towed").alias("towed"),
        F.max("police_rpt").alias("police_rpt"),
        F.max("had_initials").alias("had_initials"),
        F.max("had_prefix").alias("had_prefix"),
    )
    g = (
        g.withColumn(
            "serious",
            ((F.col("injured") > 0) | (F.col("deaths") > 0) | (F.col("medical_attn") == 1)).cast("int"),
        )
        .withColumn("n_words", F.size(F.split("text", " ")))
        .withColumn("n_chars", F.length("text"))
        .withColumn("text_hash", F.sha2(F.lower("text"), 256))
        .filter(F.col("n_words") >= min_words)
    )
    # Deduplica del testo identico tra reclami diversi (copia e incolla): si tiene il piu' vecchio.
    w = Window.partitionBy("text_hash").orderBy("ldate", "cmplid")
    g = g.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn")
    return (
        g.withColumn("year", F.year("ldate"))
        .withColumn(
            "split",
            F.when(F.col("ldate") <= F.lit(train_end).cast("date"), "train")
            .when(F.col("ldate") <= F.lit(val_end).cast("date"), "validation")
            .otherwise("test"),
        )
        .select(
            "odino", "cmplid", "ldate", "year", "faildate", "manufacturer", "make", "model", "model_year",
            "components", "n_components", "text", "n_words", "n_chars", "text_hash", "cmpl_type",
            "injured", "deaths", "medical_attn", "crash", "fire", "towed", "police_rpt", "had_initials", "had_prefix",
            "serious", "split",
        )
    )
