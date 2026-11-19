"""Fixture: addestrano modelli minuscoli su dati sintetici e li salvano in una cartella temporanea con gli stessi
nomi e metadati prodotti dalle milestone 3 e 4. Cosi' i test non dipendono dai dati reali ne' da torch."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from serving import reference as ref
from serving.app import create_app

NUM = ["driv_age_c", "veh_age_c", "bonus_malus_c", "veh_power", "log_density"]
CAT = ["veh_gas", "veh_brand", "region"]
SEVERITY, TAIL = 1800.0, 1.4


def synthetic_policies(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "driv_age_c": rng.integers(18, 91, n).astype(float), "veh_age_c": rng.integers(0, 21, n).astype(float),
        "bonus_malus_c": np.minimum(rng.choice([50, 50, 50, 60, 80, 100, 120], n), 150).astype(float),
        "veh_power": rng.integers(4, 13, n).astype(float), "log_density": rng.normal(6, 1.5, n),
        "veh_gas": rng.choice(["'Diesel'", "'Regular'"], n), "veh_brand": rng.choice(["B1", "B2", "B12"], n),
        "region": rng.choice(["R24", "R11", "R82"], n)})
    lam = 0.05 * np.exp(0.02 * (df["bonus_malus_c"] - 50) - 0.01 * df["veh_age_c"] + 0.3 * (df["driv_age_c"] < 25))
    df["y"] = rng.poisson(lam) / 1.0
    return df


@pytest.fixture(scope="session")
def assets(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    df = synthetic_policies()
    prep = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), CAT)],
                             remainder="passthrough")
    hgb = HistGradientBoostingRegressor(loss="poisson", categorical_features=[True] * 3 + [False] * 5, max_iter=60,
                                        random_state=0)
    pipe = Pipeline([("prep", prep), ("model", hgb)]).fit(df[CAT + NUM], df["y"])
    joblib.dump(pipe, d / "frequency_hgb_payment.joblib")
    (d / "frequency_metadata_payment.json").write_text(json.dumps({
        "created_utc": "2026-01-01T00:00:00+00:00", "sklearn_version": "test",
        "features": {"numeric": NUM, "categorical": CAT}, "test_metrics": {"HGB (tuned)": {"gini_test": 0.3}}}))
    (d / "pricing_metadata_payment.json").write_text(json.dumps({
        "severity_model": "constant", "severity_constant_capped": SEVERITY, "tail_load": TAIL}))

    # NLP: frasi gravi e non gravi con rumore
    rng = np.random.default_rng(1)
    bad = ["the driver was injured and taken to the hospital after the crash", "the contact suffered a broken arm in the accident",
           "airbag failed to deploy and the passenger required medical treatment", "fatal crash, the driver died at the scene"]
    ok = ["the contact stated the check engine light came on while driving", "the vehicle was taken to the dealer for a software update",
          "the radio display froze and the manufacturer was notified", "the contact noticed a rattle from the rear suspension"]
    filler = "the vehicle was a model year purchased from the dealer and the mileage was approximately"
    texts, y = [], []
    for _ in range(600):
        s = rng.random() < 0.3
        texts.append(f"{rng.choice(bad if s else ok)} {filler} {rng.integers(1000, 99999)}")
        y.append(int(s))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(texts)
    clf = LogisticRegression(max_iter=500).fit(vec.transform(texts), y)
    joblib.dump({"vectorizer": vec, "classifier": clf}, d / "nlp_tfidf.joblib")
    s1 = clf.predict_proba(vec.transform(texts))[:, 1]
    words = [len(t.split(" ")) for t in texts]
    reference = {
        "pricing": {"numeric": {c: ref.numeric_reference(df[c]) for c in NUM},
                    "categorical": {c: ref.categorical_reference(df[c]) for c in CAT},
                    "mean_rate": float(pipe.predict(df[CAT + NUM]).mean()), "n_train": len(df)},
        "nlp": {"threshold_candidate": float(np.quantile(s1, 0.8)), "candidate_share": 0.2, "review_rate": 0.05,
                "blend": {"intercept": -1.0, "coef_stage1": 0.5, "coef_stage2": 1.0}, "review_threshold": 0.5,
                "words": ref.numeric_reference(words), "created_utc": "2026-01-01"}}
    (d / "serving_reference.json").write_text(json.dumps(reference))
    return {"dir": d, "df": df, "texts": texts, "y": y, "reference": reference}


class FakeStage2:
    """Stadio 2 finto: punteggio alto se il testo contiene parole di danno (non serve torch)."""
    def score(self, texts):
        words = ("injured", "hospital", "died", "broken", "medical", "fatal")
        return np.array([0.9 if any(w in t for w in words) else 0.1 for t in texts])


@pytest.fixture()
def client(assets, tmp_path):
    app = create_app(models_dir=assets["dir"], log_path=tmp_path / "pred.jsonl", stage2=FakeStage2())
    return TestClient(app)


@pytest.fixture()
def client_stage1(assets, tmp_path):
    app = create_app(models_dir=assets["dir"], log_path=tmp_path / "pred.jsonl")
    app.state.triage.stage2 = None
    return TestClient(app)


GOOD = {"driv_age": 35, "veh_age": 4, "bonus_malus": 60, "veh_power": 6, "density": 1200.0,
        "veh_gas": "Regular", "veh_brand": "B12", "region": "R24", "exposure": 1.0}
